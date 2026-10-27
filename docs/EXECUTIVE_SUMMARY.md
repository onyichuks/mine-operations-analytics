# Executive summary

**Mine operations analytics: trusted plant data, a concentrate quality soft sensor, and gold mine safety and compliance**

Onyi Chuks, Calgary, Alberta. October 2026. [Interactive dashboard](https://onyichuks.github.io/mine-operations-analytics/) · [Full report](REPORT.md) · [Machine learning note](ML_TECHNICAL_NOTE.md) · [Code](https://github.com/onyichuks/mine-operations-analytics)

## The questions

Mines run on two kinds of operational data: process data from the plant and safety records from the site. This project asks one practical question of each, using public data.

1. **Processing plant.** Six months of 20-second readings and hourly laboratory assays from an iron ore flotation plant (2017). Can the data be trusted, and can silica in the concentrate be estimated before the laboratory result arrives?
2. **Gold mines.** United States Mine Safety and Health Administration (MSHA) records for gold mines, 2000 to 2025: 589 mines reported hours, about 150 in a typical year. Where do injuries come from, and do inspection and violation records point to the mines most at risk next year?

Every model is tested on months or years it never saw, against the simple method an operation already uses.

## What was found

**Plant data**

* **Nearly one silica value in five is not a new measurement.** Of the 4,097 hours with a silica result, only 3,323 (81 percent) are confirmed as new laboratory measurements. The rest repeat the previous hour, were filled in by interpolation, or are doubtful.
* **Two air flow sensors were frozen for 460 hours.** Columns 4 and 5 repeated one exact reading for the first month of the record. In eight further hours in June and July the data feed stalled: almost every sensor froze, or a quarter or more of the rows were exact copies.
* **The usual way of testing on this dataset overstates accuracy.** Shuffling 20-second rows into training and test sets, with iron in concentrate as an input, scores R² 0.93. Iron comes from the same laboratory sample as silica, so that score is leakage. Tested on next month's measured hours, the process sensors alone explain nothing (R² -0.03).
* **Filled-in values flatter a model too.** The soft sensor's R² is 0.17 on measured hours, but 0.34 if it is trained and scored with the repeated and interpolated hours treated as real measurements.

**Soft sensor**

* **The benchmark is the latest laboratory result.** With a 2-hour laboratory turnaround, the soft sensor's mean error was 0.601 silica percentage points against 0.635 for the latest result: 5 percent lower.
* **Its value grows as results get older:** 16 percent lower error when the latest result is 6 hours old, and 20 percent at 12 hours, for example when the laboratory is late or down.
* **Its 80 percent ranges held 81 percent of the time,** calibrated each month on the month before.
* **Almost all of the gain comes from reading the laboratory history well.** Compared with a history-only model built the same way, the process sensors add nothing at 1 to 2 hours and 2 to 3 percent at 6 to 12 hours.
* **Operators react to the laboratory.** Amine flow is most closely tied to the silica result from about three hours earlier. Settings therefore carry past quality, which may be part of the reason models built on settings alone fail.

**Gold mine safety**

* **Injury rates fell until 2018 and have risen since.** The lost-time injury rate fell from 2.58 per 200,000 hours in 2000 to 0.70 in 2018, and was 0.99 in 2025.
* **Injuries have become more serious.** The median time away or restricted was 15 days in 2016 to 2020 and 24 days in 2021 to 2025, with 46 percent of cases over 30 days.
* **Mills have about the same lost-time injury rate as underground mining.** In 2016 to 2025 mills ran at 0.80 lost-time injuries per 200,000 hours, and underground mining at 0.81.
* **Frequent and fatal injuries have different causes.** Handling materials and slips or falls were 65 percent of operator lost-time injuries in 2016 to 2025. Powered haulage caused 21 of the 48 deaths (operators and contractors) from 2000 to 2025, yet is under a tenth of lost-time injuries.
* **New workers are over-represented.** A third of workers with a lost-time injury had been at the mine less than a year, and two thirds of injured contractors.

**Inspections and the injury rate model**

* **Past injuries, not citations, point to next year's injuries, but the signal is weak.** Rates fell over the period, so the baseline gives every mine last year's industry rate. Beyond that, a mine's own three-year rate relative to the industry explains 10 percent of the variation in next-year injury counts, and Poisson regression and gradient boosting explain 13 to 15 percent.
* **Inspection and violation records add nothing.** Adding them changes the boosting model's skill by 0.0 points (90 percent interval -1.9 to +2.1).
* **Ranking mines is still useful.** The fifth of mines ranked highest each year by gradient boosting had 1.8 times the average lost-time injury rate the following year, and the highest band three times the lowest.
* **Skill varies a lot from year to year.** For gradient boosting with injury history it ranged from -6 to +45 percent and was below the baseline in 2019, 2022 and 2025.
* **Seriousness ratings and injuries moved apart.** Inspectors still find about 11 violations per 100 inspection hours. Since 2018, the share judged significant and substantial fell from 17.5 to 9.7 percent while the lost-time injury rate rose from 0.70 to 0.99, which suggests a change in enforcement practice rather than in risk.

## What it means

1. **Audit plant data before modelling it.** Flag repeated, interpolated and frozen values where the data is stored, and score models only on real measurements. Otherwise a model can look far better than it is.
2. **Judge a soft sensor against the latest laboratory result,** not against R². Here it is worth running when results are late. Cutting the turnaround from 2 hours to 1 lowers the error of the latest result by 17 percent, more than the soft sensor adds at 2 hours.
3. **Put the investment where the signal is.** The process sensors carry little information about silica on their own. An online analyzer, or faster laboratory turnaround, would do more than a more complex model.
4. **Target safety work at the causes of death, not only the causes of injury counts.** Powered haulage and machinery need their own controls (traffic management, proximity detection, guarding and lockout) even though they are a small share of injuries.
5. **Focus on the first year at a mine,** especially for contractors.
6. **Use a mine's own recent injury rate, relative to the industry, to decide where to look first.** Refit it every year, track how well it ranks, and do not rely on citation counts as a leading indicator.

## How it was done

The soft sensor uses gradient boosting (scikit-learn) on hourly features, predicting the change from the latest laboratory result. It is retrained monthly, shrunk toward the latest result by a share chosen on the previous month, and given split conformal ranges.

The injury rate model is a Poisson model of next-year lost-time injuries with hours as exposure. Each mine's rate is expressed against the industry rate of the same year. It is retrained every year and scored with the Poisson deviance against last year's industry rate. Differences between models are given 90 percent intervals by resampling whole mines.

The work is a tested Python package (16 unit tests) with an interactive dashboard, a research report and a machine learning note.

## Limits

* The plant data covers one anonymous plant over six months. The flow units are undocumented, and the laboratory turnaround is an assumption, so results are reported for 1 to 12 hours.
* The 3 percent off-specification limit is illustrative. Recovery cannot be calculated without a tailings assay.
* Safety rates cover mine operator employees only, because contractor hours are not reported by mine.
* Days lost depend on claims and return-to-work practice as well as on injury severity.

Data: OpenML dataset 43311 (CC0 public domain dedication); MSHA Open Government Data (public domain). Independent project; not affiliated with or endorsed by MSHA or the data publishers.
