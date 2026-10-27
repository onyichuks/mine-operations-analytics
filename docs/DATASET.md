# About the data

All inputs are public. The raw files are not stored in this repository; `scripts/get_data.py` downloads them and checks each against the SHA-256 checksum (a fingerprint of the file's contents) recorded in `src/mineops/data.py`, and `scripts/prepare_data.py` converts them for the analysis.

## Processing plant: Quality Prediction in a Mining Process

| | |
|---|---|
| Source | OpenML dataset 43311, https://www.openml.org/d/43311 (first published on Kaggle) |
| Licence | CC0 public domain dedication |
| File | `dataset_43311.pq`, 67.9 megabytes, Parquet |
| Plant | One iron ore flotation plant (reverse flotation: silica is floated off, iron reports to the concentrate). The plant is not named. |
| Period | 10 March 2017 01:00 to 9 September 2017 23:00 |
| Rows | 737,453 readings at 20-second intervals (180 per hour) |

Columns, as renamed in the code:

| Group | Columns | Notes |
|---|---|---|
| Feed assays | iron in feed (%), silica in feed (%) | Laboratory assays of the ore going in. They change only every 9 hours (median), so they are not hourly measurements. |
| Reagents | starch flow, amine flow | Starch depresses iron; amine collects silica. |
| Ore pulp | flow, pH, density | |
| Flotation columns | air flow and froth level in each of seven columns | |
| Concentrate assays | iron in concentrate (%), silica in concentrate (%) | Hourly laboratory results, repeated on every 20-second row of the hour. |

The publisher does not document the units of the flows, so they are used as recorded and labelled "plant units". The OpenML copy stores every value as text with a decimal comma; `flotation_to_numeric` converts them.

### Checks (written by `run_plant.py` to `outputs/plant_results.json`)

* 4,097 hours have data out of 4,415 in the period. One gap of 318 hours (16 March 06:00 to 29 March 11:00) has no data at all. Two hours are incomplete (174 and 179 rows).
* No value is outside physical limits (pH 0 to 14, density 1 to 3, assays 0 to 100 percent, no negative flows or levels).
* 1,171 rows are exact duplicates. 1,049 of them fall in eight hours on 28 June, 11 July and 13 July when the data feed stalled (almost every sensor frozen, or a quarter or more of the rows exact copies). Those hours are blanked before modelling.
* The air flows of columns 4 and 5 repeat exactly the same reading for 460 hours of data, from the start of the record to 11 April 11:00, while the other columns move normally. Those values are blanked before modelling.
* Of the 4,097 hours with a silica value, 3,323 are confirmed as new measurements (constant within the hour and different from the hour before). 446 repeat the previous hour and 310 change smoothly within the hour (interpolated at 20-second steps). 18 lie exactly on a straight line between their neighbours; with values recorded to two decimals some of these may be real results, but they are left out to be safe. Only measured hours are used to train or score models, and only measured results are used as laboratory history.
* Iron and silica in the concentrate come from the same sample and have a correlation of -0.80, so iron in concentrate is never used to predict silica.

## Gold mine safety: MSHA Open Government Data

| | |
|---|---|
| Source | United States Mine Safety and Health Administration (MSHA), https://arlweb.msha.gov/OpenGovernmentData/OGIMSHA.asp |
| Licence | United States government work, public domain |
| Format | Pipe-delimited text in zip files, Latin-1 encoding, with a definition file for each |
| Snapshot | Downloaded 27 October 2026. MSHA refreshes these files every Friday, so a fresh download will give slightly different numbers; `get_data.py` reports this rather than failing (use `--strict` to fail). |

| File | Size (zip, megabytes) | Used for | Gold mine records |
|---|---|---|---|
| Mines.zip | 7.3 | Selecting gold mines (primary Standard Industrial Classification code 104100, gold ore), mine type, status and state | 3,024 mines |
| MinesProdYearly.zip | 7.1 | Employee hours and average employment by mine, subunit and year | 8,527 rows |
| Accidents.zip | 52.3 | Accident, injury and illness reports (Title 30, Code of Federal Regulations, Part 50) | 9,836 records |
| Inspections.zip | 73.8 | Inspection events, activity type and inspector hours | 15,204 inspections |
| Violations.zip | 121.0 (1.4 gigabytes unzipped, streamed) | Citations and orders, significant and substantial finding, negligence, proposed penalty | 56,331 violations |

Of the 3,024 gold mines, 70 are active today and 2,848 abandoned; 1,645 are surface mines, 1,128 underground and 250 facilities (mills). Nevada (562), California (436) and Alaska (379) have the most. Only 589 of them reported employee hours in 2000 to 2025, about 150 in a typical year; the rest reported no employment in that period (most were abandoned long before).

### Definitions used

* **Rates** are per 200,000 employee hours, as MSHA publishes them. They use mine operator employees only: contractors report their hours as company totals across all mines, so contractor injuries are counted but no contractor rate is calculated.
* **All injuries**: degree of injury codes 1 to 6 (fatal, permanent disability, days away, days away and restricted, restricted only, medical treatment without days lost).
* **Lost-time injuries (NFDL, non-fatal days lost)**: codes 2 to 5, the definition the National Institute for Occupational Safety and Health (NIOSH) uses for lost-time injuries.
* Left out: accidents without an injury (0), occupational illness (7), natural causes (8), injuries to non-employees (9), other cases (10), and 33 records with no degree recorded.
* **Subunits**: open pit (strip, quarry, open pit), underground, surface at an underground mine, mill or preparation plant, office, dredge, other.
* **Significant and substantial**: the inspector's finding that a violation is reasonably likely to lead to a reasonably serious injury or illness.

### Checks (written by `run_safety.py` to `outputs/safety_results.json`)

* No duplicate accident rows and no duplicate violation numbers.
* One operator injury out of 6,361 falls in a year in which its mine reported no hours.
* 2.5 percent of lost-time injury records have no experience at the mine recorded.
* 138 inspection records have no inspector hours.
* 2026 is partial (12.9 million hours so far), so every trend stops at 2025.

## Use

Both sources are public domain. This is an independent project and is not affiliated with or endorsed by MSHA, the plant, or the dataset publishers.
