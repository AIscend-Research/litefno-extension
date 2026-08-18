r"""Where do real disaster allocations sit in the alpha-fairness family? (ext32)

ext22 put an alpha-fair allocation layer over a surrogate's forecast and derived,
in closed form, how much a surrogate's error costs the resulting decision:
sensitivity is U-shaped in alpha with an exact zero at the envy-free point.
ext23 read the same layer as a mechanism and found the incentive to misreport is
the *same derivative*, so the only strategy-proof member of the family is the one
that ignores the state.

Both are statements about a family of rules. Neither says where any real
allocation sits in that family. This does, using FEMA Public Assistance
obligations against CDC/ATSDR social vulnerability.

The bridge is exact, not analogical
------------------------------------
The family allocates ``a ∝ g^beta`` with ``beta = (1 - alpha) / alpha``
(``litefno.allocation.allocation_exponent``). So a log-log regression of award
against need estimates beta directly, and

    alpha_hat      = 1 / (1 + beta_hat)
    manipulability = |1 - alpha| / alpha = |beta|          (ext23)
    fragility      = (1 - alpha)^2 / (2 alpha)             (ext22)

The middle line is the point. ext23's manipulation incentive **is** the fitted
elasticity -- not something like it. Fitting the slope of award against
vulnerability therefore measures, in ext23's own units, how much a county could
gain by overstating its need.

The confound this design exists to defeat
-----------------------------------------
Public Assistance obligations track *damage*, not vulnerability. A high-SVI
county with no hurricane receives nothing, so a pooled cross-section of all
counties would mostly measure where storms land and would say nothing about how
money is split among the affected.

Every fit here is therefore **within one disaster**, across only the counties
that FEMA actually declared for that disaster. All counties in a fit were hit by
the same event under the same rules, so the remaining variation in award per
capita is allocation rather than exposure. Disasters are selected by a stated
rule -- incident type, a minimum county count, a declaration-year floor -- rather
than picked after seeing their slopes.

What the estimate does and does not assume
------------------------------------------
It assumes SVI is a stand-in for the ``g`` the family allocates over. That is an
assumption about what "need" means, and it is the weakest link: SVI is a
percentile rank, so it is ordinal, and an elasticity computed on a rank is not
the same object as an elasticity computed on a cardinal gain. The rank is used
because it is the published quantity, and the sign and rough magnitude of beta
survive that objection even though its third digit does not.

Population is the exposure denominator: awards are compared per capita, so a
large county does not read as generous merely for being large.
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from litefno.allocation import (                      # noqa: E402
    allocation_exponent, fragility_coefficient)

FEMA = "https://www.fema.gov/api/open/v2"
SVI = ("https://services3.arcgis.com/ZvidGQkLaDJxRSJ2/arcgis/rest/services/"
       "CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/1/query")


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, retries: int = 4) -> dict:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120,
                                        context=_ctx()) as r:
                return json.loads(r.read())
        except Exception as exc:
            if k == retries - 1:
                raise
            print(f"      retry {k+1}: {type(exc).__name__}", flush=True)
            time.sleep(3 * (k + 1))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def fetch_svi() -> dict:
    """County FIPS -> (svi_percentile, population). CDC/ATSDR SVI 2022."""
    out, offset = {}, 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1", "outFields": "FIPS,RPL_THEMES,E_TOTPOP",
            "returnGeometry": "false", "f": "json",
            "resultOffset": offset, "resultRecordCount": 2000})
        d = _get(f"{SVI}?{q}")
        feats = d.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f["attributes"]
            fips, svi, pop = a.get("FIPS"), a.get("RPL_THEMES"), a.get("E_TOTPOP")
            # SVI uses -999 for suppressed counties; drop rather than model them
            if fips and svi is not None and pop and svi >= 0 and pop > 0:
                out[str(fips).zfill(5)] = (float(svi), float(pop))
        offset += len(feats)
        print(f"    svi {offset}", flush=True)
        if len(feats) < 2000:
            break
    return out


def fetch_declared_counties(disaster: int) -> set:
    """County FIPS declared for this disaster under Public Assistance."""
    out, skip = set(), 0
    while True:
        q = urllib.parse.urlencode({
            "$filter": f"disasterNumber eq {disaster} and paProgramDeclared eq true",
            "$select": "fipsStateCode,fipsCountyCode",
            "$top": 1000, "$skip": skip})
        d = _get(f"{FEMA}/DisasterDeclarationsSummaries?{q}")
        rows = d.get("DisasterDeclarationsSummaries", [])
        for r in rows:
            s, c = r.get("fipsStateCode"), r.get("fipsCountyCode")
            if s and c and c != "000":
                out.add(f"{str(s).zfill(2)}{str(c).zfill(3)}")
        skip += len(rows)
        if len(rows) < 1000:
            break
    return out


def fetch_awards(disaster: int) -> dict:
    """County FIPS -> total federal share obligated for this disaster."""
    out, skip = {}, 0
    while True:
        q = urllib.parse.urlencode({
            "$filter": f"disasterNumber eq {disaster}",
            "$select": "stateNumberCode,countyCode,federalShareObligated",
            "$top": 1000, "$skip": skip})
        d = _get(f"{FEMA}/PublicAssistanceFundedProjectsDetails?{q}")
        rows = d.get("PublicAssistanceFundedProjectsDetails", [])
        for r in rows:
            s, c = r.get("stateNumberCode"), r.get("countyCode")
            amt = r.get("federalShareObligated")
            if s is None or c is None or amt is None:
                continue
            fips = f"{int(s):02d}{int(c):03d}"
            out[fips] = out.get(fips, 0.0) + float(amt)
        skip += len(rows)
        if len(rows) < 1000:
            break
    return out


def fetch_ia_awards(disaster: int, page: int = 10000) -> dict:
    """County FIPS -> total IHP awarded, from Individual Assistance registrations.

    Public Assistance repairs public infrastructure; Individual Assistance pays
    households. If a vulnerability gradient exists anywhere in FEMA's response it
    should be here, so this is the program the equity claim really wants -- PA is
    the one whose null result is least surprising.

    Registrations are one row per household, so this sums rather than averages:
    a county's total IHP is what it received, and dividing by population gives
    the same per-capita quantity the PA path uses.
    """
    out, skip = {}, 0
    while True:
        q = urllib.parse.urlencode({
            "$filter": f"disasterNumber eq {disaster}",
            "$select": "fips,ihpAmount", "$top": page, "$skip": skip})
        d = _get(f"{FEMA}/IndividualsAndHouseholdsProgramValidRegistrations?{q}")
        rows = d.get("IndividualsAndHouseholdsProgramValidRegistrations", [])
        for r in rows:
            f, amt = r.get("fips"), r.get("ihpAmount")
            if not f or amt is None:
                continue
            key = str(f).zfill(5)
            out[key] = out.get(key, 0.0) + float(amt)
        skip += len(rows)
        if len(rows) < page:
            break
        if skip % 50000 == 0:
            print(f"      ia {skip}", flush=True)
    return out


def ia_registration_count(disaster: int) -> int:
    q = urllib.parse.urlencode({
        "$filter": f"disasterNumber eq {disaster}",
        "$select": "id", "$top": 1, "$inlinecount": "allpages"})
    d = _get(f"{FEMA}/IndividualsAndHouseholdsProgramValidRegistrations?{q}")
    return int(d["metadata"]["count"])


def discover_disasters(incident: str, since: int, limit: int) -> list:
    """Disaster numbers of the given incident type, newest first.

    Selection is by incident type and declaration year only -- never by anything
    computed from the awards -- so the sample cannot be chosen to produce a
    slope.
    """
    seen, skip = {}, 0
    while True:
        q = urllib.parse.urlencode({
            "$filter": (f"incidentType eq '{incident}' and fyDeclared ge {since}"
                        " and paProgramDeclared eq true"),
            "$select": "disasterNumber,declarationTitle,fyDeclared,state",
            "$top": 1000, "$skip": skip})
        d = _get(f"{FEMA}/DisasterDeclarationsSummaries?{q}")
        rows = d.get("DisasterDeclarationsSummaries", [])
        for r in rows:
            n = r.get("disasterNumber")
            if n is not None and n not in seen:
                seen[n] = (r.get("declarationTitle", ""), r.get("fyDeclared"),
                           r.get("state", ""))
        skip += len(rows)
        if len(rows) < 1000:
            break
    order = sorted(seen.items(), key=lambda kv: -kv[0])
    return [(n, *v) for n, v in order[:limit]]


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


def fit_beta(svi: np.ndarray, award_pc: np.ndarray) -> dict:
    """Elasticity of award per capita with respect to vulnerability.

    Ordinary least squares on ``log(award per capita) = beta * log(svi) + c``.
    Counties with zero obligation are dropped rather than floored: a zero is an
    absent observation on the log scale, and replacing it with an arbitrary
    epsilon would put the fit at the mercy of that choice.
    """
    ok = (svi > 0) & (award_pc > 0) & np.isfinite(svi) & np.isfinite(award_pc)
    n = int(ok.sum())
    if n < 8:
        return {"n": n, "beta": float("nan"), "r2": float("nan"),
                "se": float("nan")}
    x, y = np.log(svi[ok]), np.log(award_pc[ok])
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    beta = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else float("nan")
    resid = y - (ym + beta * (x - xm))
    ss_tot = ((y - ym) ** 2).sum()
    r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    se = (float(np.sqrt((resid ** 2).sum() / max(n - 2, 1) / sxx))
          if sxx > 0 else float("nan"))
    return {"n": n, "beta": beta, "r2": r2, "se": se}


def implied(beta: float) -> dict:
    """Map a fitted elasticity into the alpha-family's own quantities."""
    if not np.isfinite(beta) or beta <= -1:
        # alpha = 1/(1+beta) is undefined at beta = -1 and negative below it,
        # and the family is only defined for alpha >= 0
        return {"alpha": float("nan"), "manipulability": abs(beta)
                if np.isfinite(beta) else float("nan"),
                "fragility": float("nan")}
    alpha = 1.0 / (1.0 + beta)
    return {"alpha": alpha, "manipulability": abs(beta),
            "fragility": fragility_coefficient(alpha)}


def permutation_null(svi: np.ndarray, award_pc: np.ndarray, n_perm: int,
                     seed: int) -> dict:
    """Shuffle vulnerability against award; the slope should vanish.

    Without this a nonzero beta could be an artifact of the log transform or of
    the county sample, rather than a relationship between the two variables.
    """
    rng = np.random.default_rng(seed)
    obs = fit_beta(svi, award_pc)["beta"]
    if not np.isfinite(obs):
        return {"null_mean": float("nan"), "null_sd": float("nan"),
                "p_two_sided": float("nan")}
    draws = []
    for _ in range(n_perm):
        draws.append(fit_beta(rng.permutation(svi), award_pc)["beta"])
    draws = np.array([d for d in draws if np.isfinite(d)])
    if not len(draws):
        return {"null_mean": float("nan"), "null_sd": float("nan"),
                "p_two_sided": float("nan")}
    p = float((np.abs(draws) >= abs(obs)).mean())
    return {"null_mean": float(draws.mean()), "null_sd": float(draws.std()),
            "p_two_sided": p}


# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incident-type", default="Hurricane")
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--max-disasters", type=int, default=25)
    ap.add_argument("--min-counties", type=int, default=15)
    ap.add_argument("--program", choices=("pa", "ia"), default="pa",
                    help="pa = Public Assistance (infrastructure), "
                         "ia = Individual Assistance (households)")
    ap.add_argument("--max-registrations", type=int, default=120000,
                    help="skip IA disasters larger than this, and say so")
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    args = ap.parse_args()

    print("fetching CDC/ATSDR SVI 2022 counties ...", flush=True)
    svi_map = fetch_svi()
    print(f"  {len(svi_map)} counties with SVI and population")

    print(f"discovering {args.incident_type} disasters since FY{args.since} ...",
          flush=True)
    cands = discover_disasters(args.incident_type, args.since,
                               args.max_disasters)
    print(f"  {len(cands)} candidates")

    rows, per_county = [], []
    for num, title, fy, state in cands:
        declared = fetch_declared_counties(num)
        if len(declared) < args.min_counties:
            print(f"  DR-{num} {title[:28]:30s} skip ({len(declared)} counties)",
                  flush=True)
            continue
        if args.program == "ia":
            n_reg = ia_registration_count(num)
            if n_reg > args.max_registrations:
                print(f"  DR-{num} {title[:28]:30s} skip "
                      f"({n_reg} registrations > cap)", flush=True)
                continue
            awards = fetch_ia_awards(num)
        else:
            awards = fetch_awards(num)
        fips = sorted(declared & set(svi_map))
        s = np.array([svi_map[f][0] for f in fips])
        pop = np.array([svi_map[f][1] for f in fips])
        amt = np.array([awards.get(f, 0.0) for f in fips])
        pc = np.divide(amt, pop, out=np.zeros_like(amt), where=pop > 0)

        fit = fit_beta(s, pc)
        if fit["n"] < args.min_counties:
            print(f"  DR-{num} {title[:28]:30s} skip ({fit['n']} usable)",
                  flush=True)
            continue
        null = permutation_null(s, pc, args.n_perm, args.seed)
        imp = implied(fit["beta"])
        rows.append({"disaster": num, "title": title[:48], "fy": fy,
                     "state": state, "n_declared": len(declared),
                     "total_obligated": float(amt.sum()), **fit, **imp, **null})
        for f, a, b, c in zip(fips, s, pop, pc):
            per_county.append({"disaster": num, "fips": f, "svi": a,
                               "population": b, "award_per_capita": c})
        print(f"  DR-{num} {title[:28]:30s} n={fit['n']:3d} "
              f"beta={fit['beta']:+.3f} r2={fit['r2']:.3f} "
              f"alpha={imp['alpha']:.3f} p={null['p_two_sided']:.3f}",
              flush=True)

    if not rows:
        raise SystemExit("no disasters met the criteria")

    b = np.array([r["beta"] for r in rows])
    a = np.array([r["alpha"] for r in rows], dtype=float)
    print(f"\n=== {len(rows)} {args.incident_type} disasters since FY{args.since} ===")
    print(f"    beta  : median {np.median(b):+.3f}  mean {b.mean():+.3f}  "
          f"range {b.min():+.3f} to {b.max():+.3f}")
    print(f"    positive beta in {(b > 0).sum()}/{len(b)} disasters "
          f"(award per capita rising with vulnerability)")
    fin = a[np.isfinite(a)]
    if len(fin):
        print(f"    alpha : median {np.median(fin):.3f}  "
              f"(alpha=1 is envy-free, strategy-proof, zero fragility)")
    print(f"    manipulability |beta| : median {np.median(np.abs(b)):.3f}  "
          "(ext23: max gain from overstating need)")
    sig = [r for r in rows if np.isfinite(r["p_two_sided"])
           and r["p_two_sided"] < 0.05]
    print(f"    slope beats its own permutation null in {len(sig)}/{len(rows)}")

    tag = args.program
    write_csv(args.out_dir / f"ext32_disaster_fits_{tag}.csv", rows)
    write_csv(args.out_dir / f"ext32_county_panel_{tag}.csv", per_county)


if __name__ == "__main__":
    sys.exit(main())
