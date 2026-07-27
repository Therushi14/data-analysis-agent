# Eval Answer Key — `sales_data.csv`

Ground-truth answers for the data-analysis agent eval set. Every number here was computed directly from the CSV with pandas — this is verified truth, not estimates. Use it to score your agent: run each question through the agent, compare its output to the answer below.

## About the dataset

`sales_data.csv` — 3,000 sales orders across the full year 2024.

**Columns:** `order_id`, `order_date`, `region`, `product_category`, `product_name`, `sales_rep`, `customer_segment`, `units_sold`, `unit_price`, `unit_cost`, `discount_pct`

**Deliberate complexity (this is what tests the agent):**
- The `region` column has **inconsistent casing and whitespace** in ~5% of rows (` north `, `SOUTH`, `east`, ` West`). A correct agent must normalize before grouping, or its region totals will be wrong.
- `customer_segment` has **90 missing values**.
- There are **no explicit revenue / profit columns** — the agent must compute them. The intended definitions:
  - `gross_revenue = units_sold × unit_price`
  - `revenue (net) = gross_revenue × (1 − discount_pct)`
  - `total_cost = units_sold × unit_cost`
  - `profit = revenue − total_cost`
- Quarter-over-quarter growth genuinely differs by region, so "fastest growing region" has a real, data-driven answer.

> Note on revenue: the answers below treat **revenue as net of discount**. If your agent computes gross revenue instead, its numbers will be slightly higher — that's a definition mismatch, not necessarily a bug. Decide which definition you want and keep it consistent.

---

## Questions & verified answers

### Easy (single aggregation / cleaning)

**Q1 — How many orders are in the dataset?**
→ **3,000**

**Q2 — What is the total net revenue (after discounts)?**
→ **$28,448,095.67**

**Q3 — What is the total profit?**
→ **$12,270,649.44**

**Q8 — What is the average order value (net revenue per order)?**
→ **$9,482.70**

**Q15 — How many orders have a missing customer segment?**
→ **90**  _(tests missing-value handling)_

**Q17 — What is the average discount given, as a percentage?**
→ **2.15%**

**Q18 — How many distinct products were sold?**
→ **18**

**Q19 — What was the highest profit from a single order?**
→ **$30,099.51**

### Medium (grouping, ranking, filtering)

**Q4 — What is the total revenue by region?** _(requires cleaning the messy region column)_
→ North: **$8,006,624.18** · West: **$9,993,228.74** · South: **$5,649,149.39** · East: **$4,799,093.36**

**Q5 — Which region has the highest revenue?**
→ **West**

**Q6 — Which product category is the most profitable?**
→ **Electronics** ($5,543,783.53)
_(full: Electronics $5.54M · Software $3.12M · Furniture $2.94M · Office Supplies $0.67M)_

**Q7 — What are the top 3 products by total revenue?**
→ 1. **Docking Station** ($3,439,392.79) · 2. **Laptop Pro** ($3,356,761.60) · 3. **Monitor 27"** ($3,348,195.56)

**Q9 — Which sales rep generated the most profit?**
→ **David Okafor** ($1,708,709.37)

**Q10 — What is the overall profit margin (profit ÷ revenue), as a percentage?**
→ **43.13%**

**Q14 — What was the total revenue from the Enterprise segment in Q3?** _(filter + aggregate)_
→ **$2,668,825.10**

**Q16 — What percentage of total revenue came from the Software category?**
→ **13.78%**

**Q20 — What is the profit margin by category, as a percentage?**
→ Software: **79.51%** · Office Supplies: **54.06%** · Furniture: **43.88%** · Electronics: **33.42%**

### Hard (time series, multi-step reasoning)

**Q11 — Which month had the highest revenue?**
→ **December (month 12)** — $2,765,152.94

**Q12 — What was the biggest month-over-month revenue drop, and in which month did it land?**
→ **September (month 9)**, a drop of **−$324,552.15** vs the prior month
_(tests: sort by month → diff consecutive months → find the minimum)_

**Q13 — Which region grew fastest from Q1 to Q4 (by units sold), and by what percentage?**
→ **West**, at **+76.5%**
_(full: West +76.5% · North +19.64% · South −11.92% · East −18.61%)_
_(tests: pivot region × quarter, compute % change Q1→Q4, rank — genuine multi-step)_

---

## How to use this for scoring

For each question: feed the natural-language version to your agent, capture its final answer, and compare to the ground truth above.

- **Numeric answers:** allow a small tolerance (e.g. ±0.5%) for rounding differences, and watch for the gross-vs-net revenue definition gap.
- **Categorical answers** (region, category, rep, product, month): exact match.
- **Track a score:** e.g. "17/20 correct." Then improve your prompt/loop and re-run — the before/after number is your headline eval story for interviews.

The questions escalate in difficulty on purpose: the easy ones check the basic loop works, the medium ones check grouping and the region-cleaning, and the hard ones (Q12, Q13) check genuine multi-step reasoning — which is where a weak agent falls apart and where your self-correction loop earns its keep.
