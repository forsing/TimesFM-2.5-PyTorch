#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
LOTO 7/39 — GOOGLE TIMESFM 2.5 MAKSIMALNI ZERO-SHOT SISTEM

Zvanični TimesFM 2.5 interfejs je PyTorch implementacija.
PyTorch backend. 

Za svaki CSV zasebno koristi:

1. 39 binarnih vremenskih serija pojavljivanja brojeva;
2. 39 vremenskih serija gapova;
3. 39 vremenski zaglađenih distribucijskih serija;
4. kontekste od 256, 512, 1024 i 2048 izvlačenja;
5. TimesFM tačkastu prognozu;
6. q10–q90 kvantilnu neizvesnost;
7. kalibraciju verovatnoća na očekivani zbir sedam;
8. punu expanding walk-forward proveru;
9. učenje težina samo iz ranijih walk-forward foldova;
10. potpuno zamrznuti završni holdout;
11. završno učenje težina bez korišćenja holdouta;
12. jednu NEXT predikciju za Loto;
13. jednu NEXT predikciju za Loto Plus.

TimesFM ostaje zero-shot. Njegovi parametri se ne obučavaju.
Obučavaju se samo završne težine TimesFM kandidata, koristeći
isključivo walk-forward rezultate pre završnog holdouta.
"""

import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit

try:
    import timesfm
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje biblioteka timesfm.\n\n"
        "Instalacija zvanične Google implementacije:\n\n"
        "python3 -m pip install "
        "\"timesfm @ git+https://github.com/"
        "google-research/timesfm.git\"\n"
    ) from greska


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

TEORIJSKA_STOPA = (
    BROJEVA_U_KOMBINACIJI
    / BROJ_KUGLICA
)

TEORIJSKO_OCEKIVANJE_POGODAKA = (
    BROJEVA_U_KOMBINACIJI ** 2
    / BROJ_KUGLICA
)

OCEKIVANI_GAP = (
    BROJ_KUGLICA
    / BROJEVA_U_KOMBINACIJI
)

UKUPNO_MOGUCIH_KOMBINACIJA = math.comb(
    BROJ_KUGLICA,
    BROJEVA_U_KOMBINACIJI,
)

LOTO_CSV = Path(
    "/data/"
    "loto7_4684_k73_loto_2964.csv"
)

LOTO_PLUS_CSV = Path(
    "/data/"
    "loto7_4684_k73_loto_plus_1720.csv"
)

MODEL_ID = (
    "google/timesfm-2.5-200m-pytorch"
)

KONTEKSTI = (
    256,
    512,
    1024,
    2048,
)

MAKSIMALNI_KONTEKST = max(
    KONTEKSTI
)

HORIZONT = 1

BROJ_WALK_FORWARD_FOLDOVA = 5
KORAKA_PO_WALK_FORWARD_FOLDU = 20

BROJ_HOLDOUT_KORAKA = 100
MINIMALNI_KONTEKST = 128

PROZOR_ZAGLADJIVANJA = 50
ALFA_ZAGLADJIVANJA = 2.0 / (
    PROZOR_ZAGLADJIVANJA + 1.0
)

KAZNA_NEIZVESNOSTI = 0.20

VELICINA_PAKETA = 936
BROJ_BOOTSTRAP_PONAVLJANJA = 2000

EPSILON = 1e-9

warnings.filterwarnings("ignore")

np.random.seed(SEED)
random.seed(SEED)


# =============================================================================
# ISPIS
# =============================================================================

def naslov(
    tekst: str,
    znak: str = "=",
) -> None:
    print()
    print(znak * 88)
    print(tekst)
    print(znak * 88)


def status(
    naziv: str,
    prosao: bool,
    dodatak: str = "",
) -> None:
    oznaka = (
        "PROŠLO"
        if prosao
        else "NIJE PROŠLO"
    )

    if dodatak:
        print(
            f"{naziv:<52}"
            f"{oznaka:<15}"
            f"{dodatak}"
        )
    else:
        print(
            f"{naziv:<52}"
            f"{oznaka}"
        )


def formatiraj_kombinaciju(
    kombinacija: list[int],
) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in sorted(kombinacija)
    )


# =============================================================================
# RANG KOMBINACIJE
# =============================================================================

def rang_kombinacije(
    kombinacija: list[int],
) -> int:
    kombinacija = sorted(
        int(broj)
        for broj in kombinacija
    )

    if len(kombinacija) != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            "Kombinacija mora sadržati sedam brojeva."
        )

    if len(set(kombinacija)) != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            "Brojevi moraju biti različiti."
        )

    if (
        kombinacija[0] < 1
        or kombinacija[-1] > BROJ_KUGLICA
    ):
        raise ValueError(
            "Brojevi moraju biti u opsegu 1–39."
        )

    rang = 0
    prethodni = 0
    preostalo = BROJEVA_U_KOMBINACIJI

    for broj in kombinacija:
        for kandidat in range(
            prethodni + 1,
            broj,
        ):
            rang += math.comb(
                BROJ_KUGLICA - kandidat,
                preostalo - 1,
            )

        prethodni = broj
        preostalo -= 1

    return int(rang)


# =============================================================================
# UČITAVANJE CSV PODATAKA
# =============================================================================

def ucitaj_csv(
    putanja: Path,
) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(
            f"CSV fajl ne postoji: {putanja}"
        )

    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.dropna(
        axis=0,
        how="all",
    )

    okvir = okvir.dropna(
        axis=1,
        how="all",
    )

    if okvir.shape[1] != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            f"CSV mora imati tačno sedam kolona. "
            f"Pronađeno je {okvir.shape[1]}."
        )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if okvir.isna().any().any():
        raise ValueError(
            "CSV sadrži prazne ili nenumeričke vrednosti."
        )

    izvlacenja = np.sort(
        okvir.to_numpy(
            dtype=np.int16
        ),
        axis=1,
    )

    if np.any(
        (izvlacenja < 1)
        | (izvlacenja > BROJ_KUGLICA)
    ):
        raise ValueError(
            "Svi brojevi moraju biti u opsegu 1–39."
        )

    if np.any(
        np.diff(
            izvlacenja,
            axis=1,
        ) == 0
    ):
        raise ValueError(
            "Jedno ili više izvlačenja sadrži "
            "ponovljen broj."
        )

    potreban_broj_redova = (
        MINIMALNI_KONTEKST
        + KORAKA_PO_WALK_FORWARD_FOLDU
        + BROJ_HOLDOUT_KORAKA
    )

    if len(izvlacenja) < potreban_broj_redova:
        raise ValueError(
            f"CSV mora imati najmanje "
            f"{potreban_broj_redova} redova."
        )

    return izvlacenja


# =============================================================================
# TRI PRIKAZA ISTORIJE
# =============================================================================

def napravi_binarne_serije(
    izvlacenja: np.ndarray,
) -> np.ndarray:
    binarna = np.zeros(
        (
            len(izvlacenja),
            BROJ_KUGLICA,
        ),
        dtype=np.float32,
    )

    redovi = np.repeat(
        np.arange(len(izvlacenja)),
        BROJEVA_U_KOMBINACIJI,
    )

    kolone = (
        izvlacenja.reshape(-1) - 1
    )

    binarna[
        redovi,
        kolone,
    ] = 1.0

    return binarna


def napravi_gap_serije(
    binarna: np.ndarray,
) -> np.ndarray:
    gap = np.zeros_like(
        binarna,
        dtype=np.float32,
    )

    trenutno = np.full(
        BROJ_KUGLICA,
        OCEKIVANI_GAP,
        dtype=np.float32,
    )

    for indeks in range(
        len(binarna)
    ):
        trenutno += 1.0

        pojavili_se = (
            binarna[indeks] > 0.5
        )

        trenutno[
            pojavili_se
        ] = 0.0

        gap[indeks] = trenutno

    return gap


def napravi_zagladjene_serije(
    binarna: np.ndarray,
) -> np.ndarray:
    zagladjeno = np.zeros_like(
        binarna,
        dtype=np.float32,
    )

    stanje = np.full(
        BROJ_KUGLICA,
        TEORIJSKA_STOPA,
        dtype=np.float64,
    )

    for indeks in range(
        len(binarna)
    ):
        stanje = (
            ALFA_ZAGLADJIVANJA
            * binarna[indeks]
            + (
                1.0
                - ALFA_ZAGLADJIVANJA
            )
            * stanje
        )

        zagladjeno[indeks] = stanje

    return zagladjeno


def napravi_prikaze(
    izvlacenja: np.ndarray,
) -> dict[str, np.ndarray]:
    binarna = napravi_binarne_serije(
        izvlacenja
    )

    return {
        "Binarno": binarna,
        "Gap": napravi_gap_serije(
            binarna
        ),
        "Zaglađena distribucija":
            napravi_zagladjene_serije(
                binarna
            ),
    }


# =============================================================================
# TIMESFM 2.5
# =============================================================================

def ucitaj_model():
    print(
        f"Učitavanje TimesFM 2.5 modela: "
        f"{MODEL_ID}",
        flush=True,
    )

    model = (
        timesfm.TimesFM_2p5_200M_torch
        .from_pretrained(
            MODEL_ID
        )
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=MAKSIMALNI_KONTEKST,
            max_horizon=HORIZONT,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    return model


def timesfm_prognoza(
    model,
    serije: list[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sve_tackaste = []
    svi_q10 = []
    svi_q90 = []

    ukupno = len(serije)

    for pocetak in range(
        0,
        ukupno,
        VELICINA_PAKETA,
    ):
        kraj = min(
            pocetak + VELICINA_PAKETA,
            ukupno,
        )

        tackaste, kvantili = model.forecast(
            horizon=HORIZONT,
            inputs=serije[pocetak:kraj],
        )

        tackaste = np.asarray(
            tackaste,
            dtype=np.float64,
        )

        kvantili = np.asarray(
            kvantili,
            dtype=np.float64,
        )

        sve_tackaste.append(
            tackaste[:, 0]
        )

        svi_q10.append(
            kvantili[:, 0, 1]
        )

        svi_q90.append(
            kvantili[:, 0, 9]
        )

        print(
            f"  TimesFM paket: "
            f"{kraj:,}/{ukupno:,}",
            flush=True,
        )

    tackaste = np.concatenate(
        sve_tackaste
    )

    q10 = np.concatenate(
        svi_q10
    )

    q90 = np.concatenate(
        svi_q90
    )

    q10, q90 = (
        np.minimum(q10, q90),
        np.maximum(q10, q90),
    )

    return tackaste, q10, q90


# =============================================================================
# KANDIDATI IZ SVIH PRIKAZA I KONTEKSTA
# =============================================================================

def nazivi_kandidata() -> list[str]:
    return [
        f"{prikaz} — kontekst {kontekst}"
        for kontekst in KONTEKSTI
        for prikaz in (
            "Binarno",
            "Gap",
            "Zaglađena distribucija",
        )
    ]


def sirovi_u_skor(
    prikaz: str,
    tackasta: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> np.ndarray:
    sirina = np.maximum(
        q90 - q10,
        0.0,
    )

    if prikaz == "Gap":
        osnovni_skor = np.exp(
            -np.maximum(
                tackasta,
                0.0,
            ) / OCEKIVANI_GAP
        )

        kazna = (
            KAZNA_NEIZVESNOSTI
            * sirina
            / OCEKIVANI_GAP
        )

        return np.maximum(
            osnovni_skor - kazna,
            EPSILON,
        )

    osnovni_skor = np.clip(
        tackasta,
        0.0,
        1.0,
    )

    kazna = (
        KAZNA_NEIZVESNOSTI
        * np.maximum(
            sirina,
            0.0,
        )
    )

    return np.maximum(
        osnovni_skor - kazna,
        EPSILON,
    )


def kalibrisi_na_sedam(
    skorovi: np.ndarray,
) -> np.ndarray:
    """
    Pretvara proizvoljne skorove u verovatnoće čiji je
    zbir za svaki vremenski korak tačno sedam.
    """

    skorovi = np.asarray(
        skorovi,
        dtype=np.float64,
    )

    originalni_oblik = skorovi.shape

    if skorovi.ndim == 1:
        skorovi = skorovi[
            np.newaxis,
            :
        ]

    rezultat = np.empty_like(
        skorovi
    )

    for redni, red in enumerate(
        skorovi
    ):
        sredina = float(
            np.mean(red)
        )

        standardna = float(
            np.std(red)
        )

        if standardna < EPSILON:
            logiti = np.zeros_like(
                red
            )
        else:
            logiti = (
                red - sredina
            ) / standardna

        donja = -30.0
        gornja = 30.0

        for _ in range(80):
            srednja = (
                donja + gornja
            ) / 2.0

            zbir = float(
                np.sum(
                    expit(
                        logiti + srednja
                    )
                )
            )

            if zbir > BROJEVA_U_KOMBINACIJI:
                gornja = srednja
            else:
                donja = srednja

        pomeraj = (
            donja + gornja
        ) / 2.0

        rezultat[redni] = expit(
            logiti + pomeraj
        )

    if len(originalni_oblik) == 1:
        return rezultat[0]

    return rezultat


def prognoziraj_kandidate(
    model,
    prikazi: dict[str, np.ndarray],
    trenuci: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Rezultat ima oblik:

        broj_trenutaka × 39 brojeva × broj_kandidata
    """

    imena = nazivi_kandidata()

    serije = []
    metapodaci = []

    for redni_trenutak, trenutak in enumerate(
        trenuci
    ):
        trenutak = int(
            trenutak
        )

        for kontekst in KONTEKSTI:
            pocetak = max(
                0,
                trenutak - kontekst,
            )

            for naziv_prikaza, matrica in (
                prikazi.items()
            ):
                deo = matrica[
                    pocetak:trenutak
                ]

                for broj_indeks in range(
                    BROJ_KUGLICA
                ):
                    serije.append(
                        np.asarray(
                            deo[:, broj_indeks],
                            dtype=np.float32,
                        )
                    )

                    metapodaci.append(
                        (
                            redni_trenutak,
                            naziv_prikaza,
                            kontekst,
                            broj_indeks,
                        )
                    )

    tackaste, q10, q90 = timesfm_prognoza(
        model=model,
        serije=serije,
    )

    rezultat = np.zeros(
        (
            len(trenuci),
            BROJ_KUGLICA,
            len(imena),
        ),
        dtype=np.float64,
    )

    indeks_kandidata = {
        ime: indeks
        for indeks, ime in enumerate(
            imena
        )
    }

    privremeni_skorovi = {}

    for indeks, metapodatak in enumerate(
        metapodaci
    ):
        (
            redni_trenutak,
            naziv_prikaza,
            kontekst,
            broj_indeks,
        ) = metapodatak

        kljuc = (
            redni_trenutak,
            naziv_prikaza,
            kontekst,
        )

        if kljuc not in privremeni_skorovi:
            privremeni_skorovi[kljuc] = {
                "tackasta": np.zeros(
                    BROJ_KUGLICA,
                    dtype=np.float64,
                ),
                "q10": np.zeros(
                    BROJ_KUGLICA,
                    dtype=np.float64,
                ),
                "q90": np.zeros(
                    BROJ_KUGLICA,
                    dtype=np.float64,
                ),
            }

        privremeni_skorovi[
            kljuc
        ]["tackasta"][broj_indeks] = (
            tackaste[indeks]
        )

        privremeni_skorovi[
            kljuc
        ]["q10"][broj_indeks] = q10[indeks]

        privremeni_skorovi[
            kljuc
        ]["q90"][broj_indeks] = q90[indeks]

    for kljuc, vrednosti in (
        privremeni_skorovi.items()
    ):
        (
            redni_trenutak,
            naziv_prikaza,
            kontekst,
        ) = kljuc

        ime = (
            f"{naziv_prikaza} — "
            f"kontekst {kontekst}"
        )

        kandidat_indeks = indeks_kandidata[
            ime
        ]

        sirovi_skor = sirovi_u_skor(
            prikaz=naziv_prikaza,
            tackasta=vrednosti["tackasta"],
            q10=vrednosti["q10"],
            q90=vrednosti["q90"],
        )

        rezultat[
            redni_trenutak,
            :,
            kandidat_indeks,
        ] = kalibrisi_na_sedam(
            sirovi_skor
        )

    return rezultat, imena


# =============================================================================
# PONDERISANI TIMESFM ANSAMBL
# =============================================================================

def softmax(
    vrednosti: np.ndarray,
) -> np.ndarray:
    vrednosti = (
        vrednosti
        - np.max(vrednosti)
    )

    eksponenti = np.exp(
        vrednosti
    )

    return (
        eksponenti
        / np.sum(eksponenti)
    )


def spoji_kandidate(
    kandidati: np.ndarray,
    tezine: np.ndarray,
) -> np.ndarray:
    spojeno = np.tensordot(
        kandidati,
        tezine,
        axes=([-1], [0]),
    )

    if spojeno.ndim == 1:
        return kalibrisi_na_sedam(
            spojeno
        )

    return np.asarray(
        [
            kalibrisi_na_sedam(red)
            for red in spojeno
        ],
        dtype=np.float64,
    )


def binarne_mete(
    izvlacenja: np.ndarray,
) -> np.ndarray:
    return napravi_binarne_serije(
        izvlacenja
    ).astype(np.float64)


def nauciti_tezine(
    kandidati: np.ndarray,
    mete: np.ndarray,
) -> np.ndarray:
    broj_kandidata = kandidati.shape[-1]

    pocetni_logiti = np.zeros(
        broj_kandidata,
        dtype=np.float64,
    )

    def cilj(
        logiti: np.ndarray,
    ) -> float:
        tezine = softmax(
            logiti
        )

        prognoze = spoji_kandidate(
            kandidati,
            tezine,
        )

        prognoze = np.clip(
            prognoze,
            EPSILON,
            1.0 - EPSILON,
        )

        brier = np.mean(
            (
                prognoze - mete
            ) ** 2
        )

        log_loss = -np.mean(
            mete * np.log(prognoze)
            + (
                1.0 - mete
            )
            * np.log(
                1.0 - prognoze
            )
        )

        regularizacija = (
            0.001
            * np.sum(
                (
                    tezine
                    - 1.0 / broj_kandidata
                ) ** 2
            )
        )

        return float(
            0.5 * brier
            + 0.5 * log_loss
            + regularizacija
        )

    rezultat = minimize(
        cilj,
        pocetni_logiti,
        method="L-BFGS-B",
        options={
            "maxiter": 300,
            "ftol": 1e-12,
        },
    )

    if not np.all(
        np.isfinite(rezultat.x)
    ):
        return np.full(
            broj_kandidata,
            1.0 / broj_kandidata,
            dtype=np.float64,
        )

    return softmax(
        rezultat.x
    )


# =============================================================================
# OCENJIVANJE
# =============================================================================

def oceni_prognoze(
    verovatnoce: np.ndarray,
    stvarna_izvlacenja: np.ndarray,
) -> dict:
    pogoci = []

    for skorovi, stvarni_red in zip(
        verovatnoce,
        stvarna_izvlacenja,
    ):
        poredak = sorted(
            range(BROJ_KUGLICA),
            key=lambda indeks: (
                -float(skorovi[indeks]),
                -(indeks + 1),
            ),
        )

        predikcija = set(
            indeks + 1
            for indeks in poredak[
                :BROJEVA_U_KOMBINACIJI
            ]
        )

        stvarno = set(
            int(broj)
            for broj in stvarni_red
        )

        pogoci.append(
            len(
                predikcija
                & stvarno
            )
        )

    pogoci = np.asarray(
        pogoci,
        dtype=np.int16,
    )

    if len(pogoci) == 0:
        raise ValueError(
            "Nema prognoza za ocenjivanje."
        )

    return {
        "pogoci": pogoci,
        "broj_provera": int(
            len(pogoci)
        ),
        "prosek_pogodaka": float(
            np.mean(pogoci)
        ),
        "maksimum": int(
            np.max(pogoci)
        ),
        "potpuno_tacnih": int(
            np.sum(pogoci == 7)
        ),
        "tri_plus": float(
            np.mean(pogoci >= 3)
        ),
        "cetiri_plus": float(
            np.mean(pogoci >= 4)
        ),
        "raspodela": {
            broj: int(
                np.sum(pogoci == broj)
            )
            for broj in range(8)
        },
    }


def bootstrap_interval(
    pogoci: np.ndarray,
    seed_pomeraj: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(
        SEED + seed_pomeraj
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_PONAVLJANJA,
        dtype=np.float64,
    )

    for indeks in range(
        BROJ_BOOTSTRAP_PONAVLJANJA
    ):
        uzorak = rng.choice(
            pogoci,
            size=len(pogoci),
            replace=True,
        )

        proseci[indeks] = float(
            np.mean(uzorak)
        )

    donja, gornja = np.quantile(
        proseci,
        [0.025, 0.975],
    )

    return float(donja), float(gornja)


def ispisi_metrike(
    naziv: str,
    rezultat: dict,
    interval: tuple[float, float],
) -> None:
    print()
    print(naziv)
    print("-" * 88)

    print(
        f"Broj provera:                    "
        f"{rezultat['broj_provera']:,}"
    )

    print(
        f"Prosečan broj pogodaka:          "
        f"{rezultat['prosek_pogodaka']:.6f}"
    )

    print(
        f"Slučajno očekivanje:             "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.6f}"
    )

    print(
        f"Razlika prema slučajnom:         "
        f"{rezultat['prosek_pogodaka'] - TEORIJSKO_OCEKIVANJE_POGODAKA:+.6f}"
    )

    print(
        f"Bootstrap interval 95%:          "
        f"[{interval[0]:.6f}, "
        f"{interval[1]:.6f}]"
    )

    print(
        f"Najveći broj pogodaka:           "
        f"{rezultat['maksimum']}"
    )

    print(
        f"Potpuno tačnih:                  "
        f"{rezultat['potpuno_tacnih']:,}"
    )

    print(
        f"Stopa 3+ pogodaka:               "
        f"{rezultat['tri_plus']:.1%}"
    )

    print(
        f"Stopa 4+ pogodaka:               "
        f"{rezultat['cetiri_plus']:.1%}"
    )

    print("Raspodela pogodaka:")

    for broj in range(8):
        print(
            f"  {broj} pogodaka: "
            f"{rezultat['raspodela'][broj]:,}"
        )


# =============================================================================
# EXPANDING WALK-FORWARD
# =============================================================================

def puna_walk_forward_validacija(
    model,
    prikazi: dict[str, np.ndarray],
    izvlacenja: np.ndarray,
    kraj_razvojnog_perioda: int,
    seed_pomeraj: int,
) -> dict:
    najmanji_kraj = (
        MINIMALNI_KONTEKST
        + KORAKA_PO_WALK_FORWARD_FOLDU
    )

    krajevi_foldova = np.linspace(
        najmanji_kraj,
        kraj_razvojnog_perioda,
        BROJ_WALK_FORWARD_FOLDOVA,
        dtype=int,
    )

    prethodni_kandidati = []
    prethodne_mete = []

    ocenjene_prognoze = []
    ocenjene_mete = []
    fold_rezultati = []

    imena = nazivi_kandidata()

    for redni, kraj_folda in enumerate(
        krajevi_foldova,
        start=1,
    ):
        kraj_folda = int(
            kraj_folda
        )

        pocetak_folda = max(
            MINIMALNI_KONTEKST,
            kraj_folda
            - KORAKA_PO_WALK_FORWARD_FOLDU,
        )

        trenuci = np.arange(
            pocetak_folda,
            kraj_folda,
            dtype=np.int64,
        )

        print()
        print(
            f"Walk-forward fold "
            f"{redni}/"
            f"{BROJ_WALK_FORWARD_FOLDOVA}"
        )

        print(
            f"  Istorija pre prve provere: "
            f"{pocetak_folda:,} redova"
        )

        print(
            f"  Provera redova: "
            f"{pocetak_folda + 1:,}–"
            f"{kraj_folda:,}"
        )

        kandidati, imena = prognoziraj_kandidate(
            model=model,
            prikazi=prikazi,
            trenuci=trenuci,
        )

        mete = binarne_mete(
            izvlacenja[
                pocetak_folda:kraj_folda
            ]
        )

        if prethodni_kandidati:
            kandidati_za_tezine = np.concatenate(
                prethodni_kandidati,
                axis=0,
            )

            mete_za_tezine = np.concatenate(
                prethodne_mete,
                axis=0,
            )

            tezine = nauciti_tezine(
                kandidati_za_tezine,
                mete_za_tezine,
            )
        else:
            tezine = np.full(
                len(imena),
                1.0 / len(imena),
                dtype=np.float64,
            )

        prognoze = spoji_kandidate(
            kandidati,
            tezine,
        )

        rezultat = oceni_prognoze(
            prognoze,
            izvlacenja[
                pocetak_folda:kraj_folda
            ],
        )

        rezultat["fold"] = redni
        rezultat["pocetak"] = pocetak_folda
        rezultat["kraj"] = kraj_folda
        rezultat["tezine"] = tezine.copy()

        fold_rezultati.append(
            rezultat
        )

        ocenjene_prognoze.append(
            prognoze
        )

        ocenjene_mete.append(
            izvlacenja[
                pocetak_folda:kraj_folda
            ]
        )

        prethodni_kandidati.append(
            kandidati
        )

        prethodne_mete.append(
            mete
        )

        print(
            f"  Prosek pogodaka: "
            f"{rezultat['prosek_pogodaka']:.6f}"
        )

        print(
            f"  Najviše pogodaka: "
            f"{rezultat['maksimum']}"
        )

        print(
            f"  Stopa 3+: "
            f"{rezultat['tri_plus']:.1%}"
        )

        print(
            f"  Stopa 4+: "
            f"{rezultat['cetiri_plus']:.1%}"
        )

    svi_kandidati = np.concatenate(
        prethodni_kandidati,
        axis=0,
    )

    sve_mete = np.concatenate(
        prethodne_mete,
        axis=0,
    )

    zavrsne_tezine = nauciti_tezine(
        svi_kandidati,
        sve_mete,
    )

    sve_ocenjene_prognoze = np.concatenate(
        ocenjene_prognoze,
        axis=0,
    )

    sva_ocenjena_izvlacenja = np.concatenate(
        ocenjene_mete,
        axis=0,
    )

    zbirni_rezultat = oceni_prognoze(
        sve_ocenjene_prognoze,
        sva_ocenjena_izvlacenja,
    )

    zbirni_rezultat["foldovi"] = (
        fold_rezultati
    )

    zbirni_rezultat["broj_foldova"] = (
        len(fold_rezultati)
    )

    zbirni_rezultat["tezine"] = (
        zavrsne_tezine
    )

    zbirni_rezultat["imena"] = imena

    zbirni_rezultat["bootstrap"] = (
        bootstrap_interval(
            zbirni_rezultat["pogoci"],
            seed_pomeraj=seed_pomeraj,
        )
    )

    return zbirni_rezultat


# =============================================================================
# OBRADA JEDNE IGRE
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: Path,
    model,
    seed_pomeraj: int,
) -> dict:
    pocetak_obrade = time.time()

    naslov(f"OBRADA: {naziv}")

    izvlacenja = ucitaj_csv(
        putanja
    )

    prikazi = napravi_prikaze(
        izvlacenja
    )

    holdout_pocetak = (
        len(izvlacenja)
        - BROJ_HOLDOUT_KORAKA
    )

    print(f"CSV: {putanja}")
    print(f"Broj redova: {len(izvlacenja):,}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")

    print(
        f"TimesFM prikaza po broju: "
        f"{len(prikazi)}"
    )

    print(
        f"TimesFM konteksti: "
        f"{', '.join(str(x) for x in KONTEKSTI)}"
    )

    print(
        f"Ukupno TimesFM kandidata: "
        f"{len(nazivi_kandidata())}"
    )

    print(
        f"Walk-forward foldova: "
        f"{BROJ_WALK_FORWARD_FOLDOVA}"
    )

    print(
        f"Koraka po foldu: "
        f"{KORAKA_PO_WALK_FORWARD_FOLDU}"
    )

    print(
        f"Razvojni period: "
        f"{holdout_pocetak:,} redova"
    )

    print(
        f"Zamrznuti holdout: "
        f"{BROJ_HOLDOUT_KORAKA:,} redova"
    )

    naslov("1. PUNA EXPANDING WALK-FORWARD VALIDACIJA")

    walk_forward = puna_walk_forward_validacija(
        model=model,
        prikazi=prikazi,
        izvlacenja=izvlacenja,
        kraj_razvojnog_perioda=holdout_pocetak,
        seed_pomeraj=seed_pomeraj,
    )

    ispisi_metrike(
        naziv="ZBIRNI WALK-FORWARD REZULTAT",
        rezultat=walk_forward,
        interval=walk_forward["bootstrap"],
    )

    naslov("2. TEŽINE TIMESFM KANDIDATA")

    poredak_tezina = sorted(
        range(len(walk_forward["imena"])),
        key=lambda indeks: (
            -float(
                walk_forward["tezine"][indeks]
            ),
            walk_forward["imena"][indeks],
        ),
    )

    for indeks in poredak_tezina:
        print(
            f"{walk_forward['imena'][indeks]:<48}"
            f"{walk_forward['tezine'][indeks]:.9f}"
        )

    naslov("3. ZAMRZNUTI ZAVRŠNI HOLDOUT")

    holdout_trenuci = np.arange(
        holdout_pocetak,
        len(izvlacenja),
        dtype=np.int64,
    )

    holdout_kandidati, _ = (
        prognoziraj_kandidate(
            model=model,
            prikazi=prikazi,
            trenuci=holdout_trenuci,
        )
    )

    holdout_prognoze = spoji_kandidate(
        holdout_kandidati,
        walk_forward["tezine"],
    )

    holdout = oceni_prognoze(
        holdout_prognoze,
        izvlacenja[
            holdout_pocetak:
        ],
    )

    holdout_interval = bootstrap_interval(
        holdout["pogoci"],
        seed_pomeraj=seed_pomeraj + 1000,
    )

    ispisi_metrike(
        naziv="ZAMRZNUTI ZAVRŠNI HOLDOUT",
        rezultat=holdout,
        interval=holdout_interval,
    )

    naslov("4. TIMESFM NEXT")

    next_trenutak = np.asarray(
        [len(izvlacenja)],
        dtype=np.int64,
    )

    next_kandidati, _ = prognoziraj_kandidate(
        model=model,
        prikazi=prikazi,
        trenuci=next_trenutak,
    )

    next_verovatnoce = spoji_kandidate(
        next_kandidati,
        walk_forward["tezine"],
    )[0]

    poredak = sorted(
        range(BROJ_KUGLICA),
        key=lambda indeks: (
            -float(next_verovatnoce[indeks]),
            -(indeks + 1),
        ),
    )

    next_kombinacija = sorted(
        indeks + 1
        for indeks in poredak[
            :BROJEVA_U_KOMBINACIJI
        ]
    )

    print(
        f"NEXT: "
        f"{formatiraj_kombinaciju(next_kombinacija)}"
    )

    print()
    print(
        f"{'Rang':>6}"
        f"{'Broj':>8}"
        f"{'Verovatnoća':>18}"
        f"{'Odnos prema osnovi':>24}"
    )
    print("-" * 56)

    for redni, indeks in enumerate(
        poredak,
        start=1,
    ):
        odnos = (
            next_verovatnoce[indeks]
            / TEORIJSKA_STOPA
        )

        print(
            f"{redni:>6}"
            f"{indeks + 1:>8}"
            f"{next_verovatnoce[indeks]:>17.8%}"
            f"{odnos:>24.6f}"
        )

    naslov("KONTROLNA LISTA", znak="#")

    status(
        "CSV hronološki učitan",
        len(izvlacenja) > 0,
        f"redova={len(izvlacenja):,}",
    )

    status(
        "Formirano 39 binarnih serija",
        prikazi["Binarno"].shape[1]
        == BROJ_KUGLICA,
    )

    status(
        "Formirano 39 gap serija",
        prikazi["Gap"].shape[1]
        == BROJ_KUGLICA,
    )

    status(
        "Formirano 39 zaglađenih distribucija",
        prikazi[
            "Zaglađena distribucija"
        ].shape[1] == BROJ_KUGLICA,
    )

    status(
        "Četiri TimesFM konteksta",
        len(KONTEKSTI) == 4,
    )

    status(
        "TimesFM 2.5 bez dodatne obuke",
        True,
    )

    status(
        "Walk-forward učenje težina",
        (
            len(walk_forward["tezine"])
            == len(nazivi_kandidata())
            and abs(
                float(
                    np.sum(
                        walk_forward["tezine"]
                    )
                ) - 1.0
            ) < 1e-8
        ),
    )

    status(
        "Puna expanding walk-forward validacija",
        (
            walk_forward["broj_foldova"]
            == BROJ_WALK_FORWARD_FOLDOVA
        ),
        (
            f"foldova={walk_forward['broj_foldova']}, "
            f"provera={walk_forward['broj_provera']:,}, "
            f"prosek={walk_forward['prosek_pogodaka']:.4f}"
        ),
    )

    status(
        "Zamrznuti završni holdout",
        (
            holdout["broj_provera"]
            == BROJ_HOLDOUT_KORAKA
        ),
        (
            f"provera={holdout['broj_provera']:,}, "
            f"prosek={holdout['prosek_pogodaka']:.4f}"
        ),
    )

    status(
        "Kalibracija na zbir sedam",
        abs(
            float(
                np.sum(next_verovatnoce)
            ) - BROJEVA_U_KOMBINACIJI
        ) < 1e-6,
        (
            f"zbir="
            f"{np.sum(next_verovatnoce):.6f}"
        ),
    )

    status(
        "Jedna NEXT predikcija",
        (
            len(next_kombinacija)
            == BROJEVA_U_KOMBINACIJI
            and len(set(next_kombinacija))
            == BROJEVA_U_KOMBINACIJI
        ),
    )

    naslov(
        f"KONAČNI REZULTAT — {naziv}",
        znak="#",
    )

    print(
        f"Broj redova:                      "
        f"{len(izvlacenja):,}"
    )

    print(
        f"Razvojni period:                  "
        f"{holdout_pocetak:,}"
    )

    print(
        f"Walk-forward foldova:             "
        f"{walk_forward['broj_foldova']}"
    )

    print(
        f"Walk-forward provera:             "
        f"{walk_forward['broj_provera']:,}"
    )

    print(
        f"Walk-forward prosek pogodaka:     "
        f"{walk_forward['prosek_pogodaka']:.6f}"
    )

    print(
        f"Walk-forward interval 95%:        "
        f"[{walk_forward['bootstrap'][0]:.6f}, "
        f"{walk_forward['bootstrap'][1]:.6f}]"
    )

    print(
        f"Završni holdout:                  "
        f"{holdout['broj_provera']:,}"
    )

    print(
        f"Holdout prosek pogodaka:          "
        f"{holdout['prosek_pogodaka']:.6f}"
    )

    print(
        f"Holdout interval 95%:             "
        f"[{holdout_interval[0]:.6f}, "
        f"{holdout_interval[1]:.6f}]"
    )

    print(
        f"NEXT rang:                        "
        f"{rang_kombinacije(next_kombinacija):,}"
    )

    print(
        f"NEXT:                             "
        f"{formatiraj_kombinaciju(next_kombinacija)}"
    )

    print(
        f"Vreme obrade:                     "
        f"{(time.time() - pocetak_obrade) / 60.0:.2f} minuta"
    )

    return {
        "naziv": naziv,
        "broj_redova": len(izvlacenja),
        "walk_forward": walk_forward,
        "holdout": holdout,
        "holdout_interval": holdout_interval,
        "next": next_kombinacija,
        "next_rang": rang_kombinacije(
            next_kombinacija
        ),
        "next_verovatnoce": next_verovatnoce,
    }


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    pocetak_programa = time.time()

    naslov(
        "LOTO 7/39 — GOOGLE TIMESFM 2.5 "
        "MAKSIMALNI ZERO-SHOT SISTEM"
    )

    print(f"Seed: {SEED}")

    print(
        f"Teorijska stopa broja: "
        f"{TEORIJSKA_STOPA:.9f}"
    )

    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.9f}"
    )

    print(
        f"Ukupno mogućih kombinacija: "
        f"{UKUPNO_MOGUCIH_KOMBINACIJA:,}"
    )

    print(
        f"TimesFM kandidata po igri: "
        f"{len(nazivi_kandidata())}"
    )

    model = ucitaj_model()

    loto = obradi_igru(
        naziv="Loto",
        putanja=LOTO_CSV,
        model=model,
        seed_pomeraj=0,
    )

    loto_plus = obradi_igru(
        naziv="Loto Plus",
        putanja=LOTO_PLUS_CSV,
        model=model,
        seed_pomeraj=10_000,
    )

    naslov(
        "KONAČNE NEXT PREDIKCIJE",
        znak="#",
    )

    print(
        f"Loto:           "
        f"{formatiraj_kombinaciju(loto['next'])}"
    )

    print(
        f"Loto rang:      "
        f"{loto['next_rang']:,}"
    )

    print()

    print(
        f"Loto Plus:      "
        f"{formatiraj_kombinaciju(loto_plus['next'])}"
    )

    print(
        f"Loto Plus rang: "
        f"{loto_plus['next_rang']:,}"
    )

    print()

    print(
        f"Ukupno vreme: "
        f"{(time.time() - pocetak_programa) / 60.0:.2f} minuta"
    )


if __name__ == "__main__":
    main()



"""
========================================================================================
LOTO 7/39 — GOOGLE TIMESFM 2.5 MAKSIMALNI ZERO-SHOT SISTEM
========================================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937
TimesFM kandidata po igri: 12
Učitavanje TimesFM 2.5 modela: google/timesfm-2.5-200m-pytorch
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

========================================================================================
OBRADA: Loto
========================================================================================
CSV: /data/loto7_4684_k73_loto_2964.csv
Broj redova: 2,964
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
TimesFM prikaza po broju: 3
TimesFM konteksti: 256, 512, 1024, 2048
Ukupno TimesFM kandidata: 12
Walk-forward foldova: 5
Koraka po foldu: 20
Razvojni period: 2,864 redova
Zamrznuti holdout: 100 redova

========================================================================================
1. PUNA EXPANDING WALK-FORWARD VALIDACIJA
========================================================================================

Walk-forward fold 1/5
  Istorija pre prve provere: 128 redova
  Provera redova: 129–148
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.500000
  Najviše pogodaka: 3
  Stopa 3+: 5.0%
  Stopa 4+: 0.0%

Walk-forward fold 2/5
  Istorija pre prve provere: 807 redova
  Provera redova: 808–827
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 0.950000
  Najviše pogodaka: 2
  Stopa 3+: 0.0%
  Stopa 4+: 0.0%

Walk-forward fold 3/5
  Istorija pre prve provere: 1,486 redova
  Provera redova: 1,487–1,506
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.400000
  Najviše pogodaka: 4
  Stopa 3+: 15.0%
  Stopa 4+: 5.0%

Walk-forward fold 4/5
  Istorija pre prve provere: 2,165 redova
  Provera redova: 2,166–2,185
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.450000
  Najviše pogodaka: 3
  Stopa 3+: 15.0%
  Stopa 4+: 0.0%

Walk-forward fold 5/5
  Istorija pre prve provere: 2,844 redova
  Provera redova: 2,845–2,864
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.200000
  Najviše pogodaka: 3
  Stopa 3+: 10.0%
  Stopa 4+: 0.0%

ZBIRNI WALK-FORWARD REZULTAT
----------------------------------------------------------------------------------------
Broj provera:                    100
Prosečan broj pogodaka:          1.300000
Slučajno očekivanje:             1.256410
Razlika prema slučajnom:         +0.043590
Bootstrap interval 95%:          [1.119750, 1.490000]
Najveći broj pogodaka:           4
Potpuno tačnih:                  0
Stopa 3+ pogodaka:               9.0%
Stopa 4+ pogodaka:               1.0%
Raspodela pogodaka:
  0 pogodaka: 24
  1 pogodaka: 32
  2 pogodaka: 35
  3 pogodaka: 8
  4 pogodaka: 1
  5 pogodaka: 0
  6 pogodaka: 0
  7 pogodaka: 0

========================================================================================
2. TEŽINE TIMESFM KANDIDATA
========================================================================================
Gap — kontekst 256                              0.359886881
Gap — kontekst 512                              0.318746507
Zaglađena distribucija — kontekst 1024          0.149194907
Zaglađena distribucija — kontekst 256           0.062405309
Zaglađena distribucija — kontekst 2048          0.052911189
Zaglađena distribucija — kontekst 512           0.023013358
Gap — kontekst 1024                             0.016116829
Binarno — kontekst 2048                         0.006807521
Binarno — kontekst 1024                         0.006806394
Gap — kontekst 2048                             0.001912776
Binarno — kontekst 512                          0.001099241
Binarno — kontekst 256                          0.001099087

========================================================================================
3. ZAMRZNUTI ZAVRŠNI HOLDOUT
========================================================================================
  TimesFM paket: 936/46,800
  TimesFM paket: 1,872/46,800
  TimesFM paket: 2,808/46,800
  TimesFM paket: 3,744/46,800
  TimesFM paket: 4,680/46,800
  TimesFM paket: 5,616/46,800
  TimesFM paket: 6,552/46,800
  TimesFM paket: 7,488/46,800
  TimesFM paket: 8,424/46,800
  TimesFM paket: 9,360/46,800
  TimesFM paket: 10,296/46,800
  TimesFM paket: 11,232/46,800
  TimesFM paket: 12,168/46,800
  TimesFM paket: 13,104/46,800
  TimesFM paket: 14,040/46,800
  TimesFM paket: 14,976/46,800
  TimesFM paket: 15,912/46,800
  TimesFM paket: 16,848/46,800
  TimesFM paket: 17,784/46,800
  TimesFM paket: 18,720/46,800
  TimesFM paket: 19,656/46,800
  TimesFM paket: 20,592/46,800
  TimesFM paket: 21,528/46,800
  TimesFM paket: 22,464/46,800
  TimesFM paket: 23,400/46,800
  TimesFM paket: 24,336/46,800
  TimesFM paket: 25,272/46,800
  TimesFM paket: 26,208/46,800
  TimesFM paket: 27,144/46,800
  TimesFM paket: 28,080/46,800
  TimesFM paket: 29,016/46,800
  TimesFM paket: 29,952/46,800
  TimesFM paket: 30,888/46,800
  TimesFM paket: 31,824/46,800
  TimesFM paket: 32,760/46,800
  TimesFM paket: 33,696/46,800
  TimesFM paket: 34,632/46,800
  TimesFM paket: 35,568/46,800
  TimesFM paket: 36,504/46,800
  TimesFM paket: 37,440/46,800
  TimesFM paket: 38,376/46,800
  TimesFM paket: 39,312/46,800
  TimesFM paket: 40,248/46,800
  TimesFM paket: 41,184/46,800
  TimesFM paket: 42,120/46,800
  TimesFM paket: 43,056/46,800
  TimesFM paket: 43,992/46,800
  TimesFM paket: 44,928/46,800
  TimesFM paket: 45,864/46,800
  TimesFM paket: 46,800/46,800

ZAMRZNUTI ZAVRŠNI HOLDOUT
----------------------------------------------------------------------------------------
Broj provera:                    100
Prosečan broj pogodaka:          1.300000
Slučajno očekivanje:             1.256410
Razlika prema slučajnom:         +0.043590
Bootstrap interval 95%:          [1.140000, 1.460250]
Najveći broj pogodaka:           3
Potpuno tačnih:                  0
Stopa 3+ pogodaka:               7.0%
Stopa 4+ pogodaka:               0.0%
Raspodela pogodaka:
  0 pogodaka: 16
  1 pogodaka: 45
  2 pogodaka: 32
  3 pogodaka: 7
  4 pogodaka: 0
  5 pogodaka: 0
  6 pogodaka: 0
  7 pogodaka: 0

========================================================================================
4. TIMESFM NEXT
========================================================================================
  TimesFM paket: 468/468
NEXT: 05, x, 14, y, 28, z, 30

  Rang    Broj       Verovatnoća      Odnos prema osnovi
--------------------------------------------------------
     1      30     63.68356869%                3.548085
     2       y     53.05025415%                2.955657
     3       x     52.83780214%                2.943820
     4       5     45.11275618%                2.513425
     5       z     42.55246893%                2.370780
     6      28     37.35428493%                2.081167
     7      14     32.74326799%                1.824268
     8      19     32.50792476%                1.811156
     9      38     28.32974729%                1.578372
    10      18     26.50730965%                1.476836
    11      10     22.22878819%                1.238461
    12      31     21.48236913%                1.196875
    13      22     16.73828554%                0.932562
    14      36     16.51039196%                0.919865
    15      11     14.56643385%                0.811558
    16      24     14.01508853%                0.780841
    17       4     13.93860457%                0.776579
    18      25     13.30769523%                0.741429
    19      12     12.94955672%                0.721475
    20      39     12.24326944%                0.682125
    21       8     10.23818089%                0.570413
    22      35      9.68400404%                0.539537
    23       9      9.23128793%                0.514315
    24      13      8.35273880%                0.465367
    25      34      8.05922410%                0.449014
    26       6      7.66360156%                0.426972
    27       1      6.88940186%                0.383838
    28      27      6.81960496%                0.379949
    29      37      6.47919805%                0.360984
    30       3      6.01895328%                0.335342
    31      32      5.71132160%                0.318202
    32      20      5.63577782%                0.313993
    33      21      5.52207337%                0.307658
    34      15      5.44648910%                0.303447
    35      26      5.26844012%                0.293527
    36      33      5.12131435%                0.285330
    37      17      5.09784876%                0.284023
    38      23      5.07892376%                0.282969
    39       2      5.02174777%                0.279783

########################################################################################
KONTROLNA LISTA
########################################################################################
CSV hronološki učitan                               PROŠLO         redova=2,964
Formirano 39 binarnih serija                        PROŠLO
Formirano 39 gap serija                             PROŠLO
Formirano 39 zaglađenih distribucija                PROŠLO
Četiri TimesFM konteksta                            PROŠLO
TimesFM 2.5 bez dodatne obuke                       PROŠLO
Walk-forward učenje težina                          PROŠLO
Puna expanding walk-forward validacija              PROŠLO         foldova=5, provera=100, prosek=1.3000
Zamrznuti završni holdout                           PROŠLO         provera=100, prosek=1.3000
Kalibracija na zbir sedam                           PROŠLO         zbir=7.000000
Jedna NEXT predikcija                               PROŠLO

########################################################################################
KONAČNI REZULTAT — Loto
########################################################################################
Broj redova:                      2,964
Razvojni period:                  2,864
Walk-forward foldova:             5
Walk-forward provera:             100
Walk-forward prosek pogodaka:     1.300000
Walk-forward interval 95%:        [1.119750, 1.490000]
Završni holdout:                  100
Holdout prosek pogodaka:          1.300000
Holdout interval 95%:             [1.140000, 1.460250]
NEXT rang:                        9,032,924
NEXT:                             05, x, 14, y, 28, z, 30
Vreme obrade:                     305.35 minuta

========================================================================================
OBRADA: Loto Plus
========================================================================================
CSV: /data/loto7_4684_k73_loto_plus_1720.csv
Broj redova: 1,720
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
TimesFM prikaza po broju: 3
TimesFM konteksti: 256, 512, 1024, 2048
Ukupno TimesFM kandidata: 12
Walk-forward foldova: 5
Koraka po foldu: 20
Razvojni period: 1,620 redova
Zamrznuti holdout: 100 redova

========================================================================================
1. PUNA EXPANDING WALK-FORWARD VALIDACIJA
========================================================================================

Walk-forward fold 1/5
  Istorija pre prve provere: 128 redova
  Provera redova: 129–148
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 0.900000
  Najviše pogodaka: 2
  Stopa 3+: 0.0%
  Stopa 4+: 0.0%

Walk-forward fold 2/5
  Istorija pre prve provere: 496 redova
  Provera redova: 497–516
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.250000
  Najviše pogodaka: 3
  Stopa 3+: 5.0%
  Stopa 4+: 0.0%

Walk-forward fold 3/5
  Istorija pre prve provere: 864 redova
  Provera redova: 865–884
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.350000
  Najviše pogodaka: 3
  Stopa 3+: 20.0%
  Stopa 4+: 0.0%

Walk-forward fold 4/5
  Istorija pre prve provere: 1,232 redova
  Provera redova: 1,233–1,252
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 1.350000
  Najviše pogodaka: 3
  Stopa 3+: 20.0%
  Stopa 4+: 0.0%

Walk-forward fold 5/5
  Istorija pre prve provere: 1,600 redova
  Provera redova: 1,601–1,620
  TimesFM paket: 936/9,360
  TimesFM paket: 1,872/9,360
  TimesFM paket: 2,808/9,360
  TimesFM paket: 3,744/9,360
  TimesFM paket: 4,680/9,360
  TimesFM paket: 5,616/9,360
  TimesFM paket: 6,552/9,360
  TimesFM paket: 7,488/9,360
  TimesFM paket: 8,424/9,360
  TimesFM paket: 9,360/9,360
  Prosek pogodaka: 0.950000
  Najviše pogodaka: 4
  Stopa 3+: 5.0%
  Stopa 4+: 5.0%

ZBIRNI WALK-FORWARD REZULTAT
----------------------------------------------------------------------------------------
Broj provera:                    100
Prosečan broj pogodaka:          1.160000
Slučajno očekivanje:             1.256410
Razlika prema slučajnom:         -0.096410
Bootstrap interval 95%:          [0.970000, 1.350000]
Najveći broj pogodaka:           4
Potpuno tačnih:                  0
Stopa 3+ pogodaka:               10.0%
Stopa 4+ pogodaka:               1.0%
Raspodela pogodaka:
  0 pogodaka: 26
  1 pogodaka: 43
  2 pogodaka: 21
  3 pogodaka: 9
  4 pogodaka: 1
  5 pogodaka: 0
  6 pogodaka: 0
  7 pogodaka: 0

========================================================================================
2. TEŽINE TIMESFM KANDIDATA
========================================================================================
Binarno — kontekst 512                          0.331654588
Binarno — kontekst 2048                         0.331585900
Binarno — kontekst 1024                         0.331530069
Gap — kontekst 256                              0.003839890
Zaglađena distribucija — kontekst 256           0.000517990
Zaglađena distribucija — kontekst 1024          0.000405823
Gap — kontekst 2048                             0.000210731
Gap — kontekst 1024                             0.000110085
Zaglađena distribucija — kontekst 512           0.000056497
Zaglađena distribucija — kontekst 2048          0.000044892
Gap — kontekst 512                              0.000043486
Binarno — kontekst 256                          0.000000050

========================================================================================
3. ZAMRZNUTI ZAVRŠNI HOLDOUT
========================================================================================
  TimesFM paket: 936/46,800
  TimesFM paket: 1,872/46,800
  TimesFM paket: 2,808/46,800
  TimesFM paket: 3,744/46,800
  TimesFM paket: 4,680/46,800
  TimesFM paket: 5,616/46,800
  TimesFM paket: 6,552/46,800
  TimesFM paket: 7,488/46,800
  TimesFM paket: 8,424/46,800
  TimesFM paket: 9,360/46,800
  TimesFM paket: 10,296/46,800
  TimesFM paket: 11,232/46,800
  TimesFM paket: 12,168/46,800
  TimesFM paket: 13,104/46,800
  TimesFM paket: 14,040/46,800
  TimesFM paket: 14,976/46,800
  TimesFM paket: 15,912/46,800
  TimesFM paket: 16,848/46,800
  TimesFM paket: 17,784/46,800
  TimesFM paket: 18,720/46,800
  TimesFM paket: 19,656/46,800
  TimesFM paket: 20,592/46,800
  TimesFM paket: 21,528/46,800
  TimesFM paket: 22,464/46,800
  TimesFM paket: 23,400/46,800
  TimesFM paket: 24,336/46,800
  TimesFM paket: 25,272/46,800
  TimesFM paket: 26,208/46,800
  TimesFM paket: 27,144/46,800
  TimesFM paket: 28,080/46,800
  TimesFM paket: 29,016/46,800
  TimesFM paket: 29,952/46,800
  TimesFM paket: 30,888/46,800
  TimesFM paket: 31,824/46,800
  TimesFM paket: 32,760/46,800
  TimesFM paket: 33,696/46,800
  TimesFM paket: 34,632/46,800
  TimesFM paket: 35,568/46,800
  TimesFM paket: 36,504/46,800
  TimesFM paket: 37,440/46,800
  TimesFM paket: 38,376/46,800
  TimesFM paket: 39,312/46,800
  TimesFM paket: 40,248/46,800
  TimesFM paket: 41,184/46,800
  TimesFM paket: 42,120/46,800
  TimesFM paket: 43,056/46,800
  TimesFM paket: 43,992/46,800
  TimesFM paket: 44,928/46,800
  TimesFM paket: 45,864/46,800
  TimesFM paket: 46,800/46,800

ZAMRZNUTI ZAVRŠNI HOLDOUT
----------------------------------------------------------------------------------------
Broj provera:                    100
Prosečan broj pogodaka:          1.210000
Slučajno očekivanje:             1.256410
Razlika prema slučajnom:         -0.046410
Bootstrap interval 95%:          [1.030000, 1.400000]
Najveći broj pogodaka:           4
Potpuno tačnih:                  0
Stopa 3+ pogodaka:               11.0%
Stopa 4+ pogodaka:               1.0%
Raspodela pogodaka:
  0 pogodaka: 25
  1 pogodaka: 41
  2 pogodaka: 23
  3 pogodaka: 10
  4 pogodaka: 1
  5 pogodaka: 0
  6 pogodaka: 0
  7 pogodaka: 0

========================================================================================
4. TIMESFM NEXT
========================================================================================
  TimesFM paket: 468/468
NEXT: 12, x, 16, y, 19, z, 27

  Rang    Broj       Verovatnoća      Odnos prema osnovi
--------------------------------------------------------
     1       z     65.04253814%                3.623799
     2       y     48.53558023%                2.704125
     3       x     40.88161395%                2.277690
     4      27     40.67023938%                2.265913
     5      19     37.29057407%                2.077618
     6      16     34.14193858%                1.902194
     7      12     30.12046705%                1.678140
     8       9     30.10374781%                1.677209
     9      24     29.07833256%                1.620079
    10      23     24.26679708%                1.352007
    11      31     24.17945601%                1.347141
    12      15     23.76212594%                1.323890
    13       2     22.24616323%                1.239429
    14      38     20.26374932%                1.128980
    15       1     19.81152282%                1.103785
    16      29     17.95004109%                1.000074
    17       8     17.88127509%                0.996242
    18      28     17.77793385%                0.990485
    19      21     17.52840997%                0.976583
    20      22     15.70652307%                0.875078
    21      39     15.06349877%                0.839252
    22      35      9.69054122%                0.539902
    23       6      7.38619110%                0.411516
    24      36      7.10022874%                0.395584
    25       4      6.90803168%                0.384876
    26      33      6.90119021%                0.384495
    27      11      6.77049725%                0.377213
    28      30      6.58199666%                0.366711
    29       7      6.46945018%                0.360441
    30      13      6.01811590%                0.335295
    31      20      5.87256599%                0.327186
    32      18      5.47461679%                0.305014
    33      34      5.10520281%                0.284433
    34      10      5.00553525%                0.278880
    35       3      4.73431358%                0.263769
    36       5      4.52701379%                0.252219
    37      37      4.47661988%                0.249412
    38      32      4.34022082%                0.241812
    39      26      4.33514012%                0.241529

########################################################################################
KONTROLNA LISTA
########################################################################################
CSV hronološki učitan                               PROŠLO         redova=1,720
Formirano 39 binarnih serija                        PROŠLO
Formirano 39 gap serija                             PROŠLO
Formirano 39 zaglađenih distribucija                PROŠLO
Četiri TimesFM konteksta                            PROŠLO
TimesFM 2.5 bez dodatne obuke                       PROŠLO
Walk-forward učenje težina                          PROŠLO
Puna expanding walk-forward validacija              PROŠLO         foldova=5, provera=100, prosek=1.1600
Zamrznuti završni holdout                           PROŠLO         provera=100, prosek=1.2100
Kalibracija na zbir sedam                           PROŠLO         zbir=7.000000
Jedna NEXT predikcija                               PROŠLO

########################################################################################
KONAČNI REZULTAT — Loto Plus
########################################################################################
Broj redova:                      1,720
Razvojni period:                  1,620
Walk-forward foldova:             5
Walk-forward provera:             100
Walk-forward prosek pogodaka:     1.160000
Walk-forward interval 95%:        [0.970000, 1.350000]
Završni holdout:                  100
Holdout prosek pogodaka:          1.210000
Holdout interval 95%:             [1.030000, 1.400000]
NEXT rang:                        14,273,599
NEXT:                             12, x, 16, y, 19, z, 27
Vreme obrade:                     305.43 minuta

########################################################################################
KONAČNE NEXT PREDIKCIJE
########################################################################################
Loto:           05, x, 14, y, 28, z, 30
Loto rang:      9,032,924

Loto Plus:      12, x, 16, y, 19, z, 27
Loto Plus rang: 14,273,599

Ukupno vreme: 610.81 minuta
"""



"""
Google TimesFM 2.5 može da se primeni na Loto CSV podatke bez obuke modela, 
ali uz pravilno predstavljanje problema.

TimesFM 2.5 je model za univarijantne vremenske serije. 
Zato se svaki od 39 brojeva pretvara u zasebnu binarnu vremensku seriju:
Broj 1:  0, 1, 0, 0, 1, 0, ...
Broj 2:  1, 0, 0, 1, 0, 0, ...
...
Broj 39: 0, 0, 1, 0, 1, 0, ...
1 znači da se broj pojavio u tom izvlačenju, a 0 da nije. TimesFM zatim, bez treniranja, prognozira narednu vrednost svake od 39 serija.

Konačan postupak bio bi:
CSV izvlačenja
→ 39 binarnih vremenskih serija
→ TimesFM 2.5 zero-shot prognoza za sledeći korak
→ kontinuirani skor za svaki broj
→ korekcija i normalizacija skorova
→ rangiranje svih 39 brojeva
→ sedam najvećih skorova
→ jedna NEXT kombinacija

Prednosti:
- ne obučavaš model na samo nekoliko hiljada redova;
- koristi unapred obučeni vremenski foundation model;
- dobijae može da vrati srednju prognozu i kvantilnu neizvesnost;
- kontekst do 16.384 tačke dovoljan je CSV-a;
- može da radi lokalno preko PyTorch verzije, uključujući CPU kada nema CUDA sistema.

Ograničenje je važno: 
TimesFM 2.5 svaku od 39 serija prognozira zasebno, 
pa sam ne modeluje veze između brojeva i ne zna da tačno sedam brojeva mora biti izabrano. 
Zato se na kraju uzima sedam najvećih skorova, a veze parova i sedmorke ne treba mu pripisivati.

Procena vrednosti bila bi: zero-shot walk-forward kroz istoriju
→ tačno isti kontekst za svaki presek
→ sedam najvećih prognoza
→ broj pogodaka
→ zaključani završni holdout
→ poređenje sa slučajnim očekivanjem 1,256410

Potpuno nov zero-shot vremenski model za proveru, 
ali tek walk-forward i holdout mogu pokazati da li je koristan za Loto podatke. 
Zvanični TimesFM 2.5 model ima 200 miliona parametara, 
univarijantni režim, kvantilne prognoze i kontekst do 16.384 tačke.

TimesFM 2.5 je univarijantan model, 
pa sam ne uči veze između svih 39 brojeva niti pravilo da tačno sedam mora biti izabrano.
"""
