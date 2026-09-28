# S04 snapshot statistics (Text Change, full cohort)

- **Generated (UTC):** 2026-09-28
- **Source:** `code_recorder_s04` (`"CodeTrace"` + `"UserAttributes"`, same logical schema as S03 under quoted CamelCase names), `identifier='Text Change'`, all 73 users with snapshots.
- **Data quality:** this database has TOAST corruption (some `content` rows unreadable — likely from an unclean shutdown); corrupt files were skipped per (user, file) with try/except and counted in the table (`skipped` column), everything else is complete.
- **Definitions:** as in `s03_snapshot_statistics.md` (final-version lines/methods via FUNC_PATTERN; span_days = last−first snapshot; ~0 span reports raw snapshot count). Scores from `S04/students_with_scores.csv`.

## Cohort summary (n=73, 57 scored)

| metric | median | mean | min–max |
|---|---|---|---|
| files | 38 | 41.2 | 1–127 |
| snapshots | 1748 | 1937.1 | 1–10561 |
| final lines | 3320 | 4778.1 | 1–28894 |
| methods | 55 | 72.2 | 0–450 |
| cpp files/user | 14 | 17.1 | 1–53 |
| hpp files/user | 15 | 16.5 | 0–59 |

## Per-student table

| student_id | score | files | cpp | hpp | other | snapshots | mean/file | max-1file | final lines | methods | span (d) | snaps/day | skipped | assignments |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 610700002 | 2.11 | 35 | 10 | 1 | 24 | 2521 | 72.0 | 506 | 3240 | 31 | 22.3 | 113.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810002010 | 0.24 | 26 | 13 | 13 | 0 | 422 | 16.2 | 173 | 2464 | 62 | 10.1 | 41.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810002023 | — | 50 | 20 | 15 | 15 | 1748 | 35.0 | 485 | 4417 | 145 | 39.4 | 44.4 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810002031 | — | 14 | 12 | 2 | 0 | 1403 | 100.2 | 1138 | 1698 | 55 | 19.2 | 73.1 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810002033 | 15.00 | 85 | 31 | 29 | 25 | 10561 | 124.2 | 5865 | 19612 | 79 | 42.8 | 246.8 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810003038 | 15.00 | 19 | 14 | 5 | 0 | 2076 | 109.3 | 1727 | 2293 | 92 | 57.9 | 35.9 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810003060 | 4.78 | 127 | 53 | 59 | 15 | 456 | 3.6 | 49 | 6922 | 91 | 49.0 | 9.3 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810101385 | 13.19 | 58 | 25 | 25 | 8 | 1090 | 18.8 | 240 | 3469 | 10 | 19.9 | 54.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810101389 | — | 28 | 13 | 15 | 0 | 1349 | 48.2 | 466 | 3072 | 19 | 17.3 | 78.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810101446 | 9.43 | 75 | 21 | 24 | 30 | 3058 | 40.8 | 577 | 8553 | 44 | 21.1 | 144.9 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810101524 | 12.21 | 28 | 13 | 12 | 3 | 2367 | 84.5 | 656 | 2026 | 81 | 61.0 | 38.8 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810102117 | — | 1 | 1 | 0 | 0 | 1 | 1.0 | 1 | 19 | 1 | 0.0 | 1 | 0 | AP-Spring04-CA6 |
| 810102596 | — | 1 | 1 | 0 | 0 | 4 | 4.0 | 4 | 1 | 0 | 6.6 | 0.6 | 0 | AP-Spring04-CA6 |
| 810103347 | 0.00 | 33 | 31 | 2 | 0 | 1280 | 38.8 | 525 | 28894 | 450 | 20.7 | 61.8 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103351 | — | 67 | 33 | 32 | 2 | 2299 | 34.3 | 360 | 5447 | 128 | 20.3 | 113.3 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103353 | — | 18 | 13 | 5 | 0 | 318 | 17.7 | 90 | 3443 | 57 | 33.4 | 9.5 | 0 | AP-Spring04-CA6-Phase3 |
| 810103363 | — | 20 | 9 | 11 | 0 | 1067 | 53.4 | 137 | 626 | 12 | 1.7 | 627.6 | 0 | AP-Spring04-CA6 |
| 810103366 | 15.00 | 117 | 50 | 50 | 17 | 3391 | 29.0 | 1096 | 24202 | 217 | 20.5 | 165.4 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103377 | 15.00 | 42 | 20 | 22 | 0 | 1341 | 31.9 | 465 | 4418 | 27 | 11.2 | 119.7 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810103382 | 2.54 | 92 | 38 | 36 | 18 | 1214 | 13.2 | 195 | 6367 | 71 | 17.3 | 70.2 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103391 | 12.52 | 10 | 4 | 6 | 0 | 1176 | 117.6 | 291 | 794 | 44 | 53.0 | 22.2 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810103396 | — | 2 | 2 | 0 | 0 | 22 | 11.0 | 13 | 2675 | 145 | 9.4 | 2.3 | 1 | AP-Spring04-CA6 |
| 810103397 | 15.00 | 31 | 10 | 10 | 11 | 3863 | 124.6 | 2190 | 6182 | 49 | 18.0 | 214.6 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103407 | — | 27 | 9 | 10 | 8 | 2052 | 76.0 | 579 | 2277 | 48 | 16.3 | 125.9 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103413 | 15.00 | 26 | 13 | 13 | 0 | 4168 | 160.3 | 802 | 2448 | 79 | 23.4 | 178.1 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103414 | 15.00 | 83 | 36 | 35 | 12 | 3824 | 46.1 | 453 | 6935 | 130 | 20.1 | 190.2 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103422 | 0.00 | 15 | 13 | 2 | 0 | 53 | 3.5 | 22 | 753 | 19 | 22.1 | 2.4 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810103424 | 15.00 | 37 | 15 | 22 | 0 | 4535 | 122.6 | 1169 | 2724 | 141 | 66.1 | 68.6 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103425 | 5.17 | 85 | 43 | 42 | 0 | 3356 | 39.5 | 260 | 1970 | 99 | 20.6 | 162.9 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103434 | 9.49 | 42 | 16 | 16 | 10 | 2153 | 51.3 | 246 | 4736 | 50 | 53.8 | 40.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103444 | — | 5 | 3 | 2 | 0 | 51 | 10.2 | 22 | 761 | 1 | 1.2 | 42.5 | 0 | AP-Spring04-CA6 |
| 810103448 | 15.00 | 38 | 18 | 20 | 0 | 1824 | 48.0 | 348 | 2036 | 37 | 17.7 | 103.1 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810103460 | 15.00 | 31 | 14 | 17 | 0 | 2644 | 85.3 | 653 | 3320 | 50 | 53.2 | 49.7 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103468 | 2.94 | 78 | 26 | 26 | 26 | 974 | 12.5 | 98 | 7465 | 54 | 60.7 | 16.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103474 | 7.74 | 68 | 13 | 17 | 38 | 2014 | 29.6 | 374 | 4746 | 61 | 23.4 | 86.1 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103476 | 15.00 | 32 | 15 | 12 | 5 | 693 | 21.7 | 118 | 2809 | 33 | 14.1 | 49.1 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103480 | 12.80 | 66 | 11 | 13 | 42 | 4018 | 60.9 | 1169 | 5250 | 63 | 24.8 | 162.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103482 | 15.00 | 29 | 12 | 17 | 0 | 1529 | 52.7 | 285 | 2273 | 59 | 58.7 | 26.0 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103485 | 0.49 | 30 | 14 | 16 | 0 | 481 | 16.0 | 185 | 1737 | 12 | 14.9 | 32.3 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2 |
| 810103490 | 15.00 | 45 | 22 | 23 | 0 | 3621 | 80.5 | 618 | 4240 | 187 | 57.0 | 63.5 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103494 | 15.00 | 44 | 16 | 15 | 13 | 4173 | 94.8 | 799 | 5805 | 125 | 19.1 | 218.5 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103496 | 11.79 | 32 | 15 | 17 | 0 | 1064 | 33.2 | 231 | 4416 | 98 | 17.6 | 60.5 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-phase3 |
| 810103498 | 0.61 | 55 | 20 | 22 | 13 | 1741 | 31.7 | 546 | 4404 | 45 | 16.7 | 104.3 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103500 | — | 23 | 10 | 13 | 0 | 2495 | 108.5 | 417 | 2020 | 37 | 23.9 | 104.4 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103504 | 15.00 | 29 | 13 | 16 | 0 | 2020 | 69.7 | 301 | 2849 | 117 | 62.0 | 32.6 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103506 | 9.95 | 45 | 14 | 16 | 15 | 2152 | 47.8 | 470 | 5695 | 46 | 21.6 | 99.6 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103510 | 2.69 | 48 | 22 | 26 | 0 | 1171 | 24.4 | 366 | 5968 | 145 | 47.9 | 24.4 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103515 | 13.38 | 49 | 19 | 16 | 14 | 2138 | 43.6 | 446 | 5158 | 47 | 20.4 | 104.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103522 | 15.00 | 50 | 21 | 20 | 9 | 2533 | 50.7 | 390 | 4782 | 29 | 59.9 | 42.3 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103525 | 6.28 | 11 | 5 | 6 | 0 | 127 | 11.5 | 48 | 1412 | 12 | 9.2 | 13.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103526 | 0.00 | 49 | 11 | 8 | 30 | 2197 | 44.8 | 622 | 7639 | 24 | 21.7 | 101.2 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103539 | — | 70 | 30 | 19 | 21 | 3882 | 55.5 | 680 | 16044 | 200 | 25.5 | 152.2 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103547 | 3.52 | 78 | 41 | 37 | 0 | 1212 | 15.5 | 204 | 6362 | 77 | 23.2 | 52.2 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103550 | — | 18 | 8 | 10 | 0 | 2289 | 127.2 | 658 | 1865 | 67 | 51.7 | 44.3 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103554 | 14.60 | 46 | 8 | 17 | 21 | 1648 | 35.8 | 242 | 3124 | 41 | 24.3 | 67.8 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103558 | 12.06 | 15 | 6 | 9 | 0 | 1452 | 96.8 | 390 | 2388 | 113 | 16.1 | 90.2 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103560 | 15.00 | 42 | 16 | 14 | 12 | 1906 | 45.4 | 626 | 6087 | 56 | 19.2 | 99.3 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103561 | 11.05 | 42 | 14 | 14 | 14 | 1872 | 44.6 | 405 | 4804 | 34 | 16.0 | 117.0 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103562 | 4.10 | 35 | 11 | 13 | 11 | 4662 | 133.2 | 1301 | 4992 | 129 | 25.2 | 185.0 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103564 | 0.86 | 13 | 7 | 6 | 0 | 1047 | 80.5 | 237 | 1833 | 69 | 22.8 | 45.9 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103567 | 3.21 | 14 | 7 | 7 | 0 | 465 | 33.2 | 93 | 499 | 44 | 18.0 | 25.8 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103568 | — | 40 | 21 | 18 | 1 | 381 | 9.5 | 45 | 2355 | 29 | 43.3 | 8.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase3 |
| 810103569 | 7.07 | 43 | 22 | 21 | 0 | 3137 | 73.0 | 518 | 4182 | 102 | 18.6 | 168.7 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103575 | 3.46 | 52 | 18 | 3 | 31 | 2273 | 43.7 | 300 | 4591 | 73 | 23.8 | 95.5 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103580 | 15.00 | 39 | 19 | 20 | 0 | 2056 | 52.7 | 447 | 2520 | 27 | 20.7 | 99.3 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103582 | 0.15 | 43 | 11 | 11 | 21 | 1476 | 34.3 | 174 | 2607 | 25 | 63.9 | 23.1 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103583 | 11.33 | 24 | 10 | 12 | 2 | 1209 | 50.4 | 467 | 1358 | 5 | 58.5 | 20.7 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103588 | 15.00 | 23 | 11 | 12 | 0 | 1795 | 78.0 | 331 | 3475 | 100 | 22.0 | 81.6 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103595 | — | 10 | 5 | 5 | 0 | 655 | 65.5 | 336 | 616 | 11 | 12.6 | 52.0 | 0 | AP-Spring04-CA6 |
| 810103597 | 14.79 | 50 | 24 | 26 | 0 | 2580 | 51.6 | 839 | 4793 | 111 | 62.1 | 41.5 | 0 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103598 | 0.00 | 58 | 22 | 30 | 6 | 1405 | 24.2 | 439 | 2987 | 25 | 21.4 | 65.7 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103599 | 15.00 | 69 | 25 | 29 | 15 | 823 | 11.9 | 184 | 20869 | 138 | 13.1 | 62.8 | 1 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |
| 810103602 | 3.49 | 32 | 15 | 17 | 0 | 355 | 11.1 | 98 | 2989 | 35 | 10.2 | 34.8 | 2 | AP-Spring04-CA6,AP-Spring04-CA6-Phase2,AP-Spring04-CA6-Phase3 |

## Behavior notes

- Same reading guide as the S03 doc (volume, grinder-vs-steady, methods/line decomposition, header discipline, burst-vs-steady spans).
- **Assignments column** is new here: S04 spans phases (AP-Spring04-CA6, -Phase2, -Phase3); students active in multiple phases vs. one phase are different populations for any cross-phase comparison.
