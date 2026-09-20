# Generated figures

All figures are vector PDF plus 300 dpi PNG, sized for an IEEE column. Error bars are 95 % bootstrap intervals over seeds.

* **fig1_deadline_miss** - Deadline-miss rate by policy and profile over 30 seeds (mean, 95% bootstrap CI over seeds).
* **fig2_p95_latency** - 95th-percentile control-output latency by policy and profile over 30 seeds.
* **fig2b_usable_confidence** - Usable confidence proxy: mean per-frame confidence actually in hand at the deadline. Synthetic proxy, not detector accuracy.
* **fig8_adaptive_baselines** - DAPPER against the adaptive baselines, all fitted on the calibration seeds only.
* **fig3_deadline_mode_distribution** - DAPPER mode share against the control deadline, pooled over the five profiles and 30 seeds.
* **fig3b_deadline_tradeoff** - DAPPER p95 control latency and deadline-miss rate against the control deadline.
* **fig4_runtime_estimation_error** - DAPPER under biased runtime estimates. Execution is unchanged; only the scheduler's beliefs are perturbed.
* **fig5_component_ablation** - One-at-a-time removal of each risk signal, with the remaining weights renormalised to sum to one.
* **fig6_scheduler_overhead** - Cost of one DapperScheduler.decide() call on the recorded workstation. Software overhead only.
* **fig7_real_detector** - Secondary real-detector validation on COCO val2017. Left: measured model quality. Right: per-frame quality each policy delivered under the replayed profiles.

`oracle*` marks the offline oracle upper bound, which is not a deployable policy.
