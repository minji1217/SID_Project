| variant | role |
|---|---|
| A | 기존: article-level Q2 nearest (r1 = h - q1) |
| B | 교수님 설계: event-level Q2(mean(h)) |
| B-r1 | 진단용: event-level Q2(mean(h - q1)) |

### SID 지표

| metric | A | B | B-r1 |
|---|---|---|---|
| counts.train_articles | 9738 | 9738 | 9738 |
| counts.validation_articles | 3122 | 3122 | 3122 |
| counts.validation_reused_train_articles | 0 | 0 | 0 |
| counts.validation_only_articles | 3122 | 3122 | 3122 |
| counts.unique_articles | 12860 | 12860 | 12860 |
| counts.train_events | 9145 | 9145 | 9145 |
| counts.train_singleton_events | 8839 | 8839 | 8839 |
| unique_codes.c1 | 25 | 25 | 25 |
| unique_codes.c2 | 128 | 128 | 128 |
| unique_codes.c3 | 512 | 512 | 512 |
| unique_codes.c12 | 1129 | 944 | 1149 |
| unique_codes.c123 | 10952 | 10761 | 10968 |
| unique_codes.c1234 | 12860 | 12860 | 12860 |
| collision.c123_collision_article_ratio | 0.2439 | 0.2666 | 0.2410 |
| collision.c123_collision_rate | 0.1484 | 0.1632 | 0.1471 |
| collision.max_c4 | 16 | 15 | 17 |
| same_event_c2.train.multi_article_events | 306 | 306 | 306 |
| same_event_c2.train.all_same_c2_event_ratio | 0.3137 | 1.0000 | 1.0000 |
| same_event_c2.train.dominant_c2_ratio_mean | 0.6640 | 1.0000 | 1.0000 |
| same_event_c2.validation.multi_article_events | 72 | 72 | 72 |
| same_event_c2.validation.all_same_c2_event_ratio | 1.0000 | 1.0000 | 1.0000 |
| same_event_c2.validation.dominant_c2_ratio_mean | 1.0000 | 1.0000 | 1.0000 |
| same_event_c2.all_unique_articles.multi_article_events | 402 | 402 | 402 |
| same_event_c2.all_unique_articles.all_same_c2_event_ratio | 0.4701 | 1.0000 | 1.0000 |
| same_event_c2.all_unique_articles.dominant_c2_ratio_mean | 0.7415 | 1.0000 | 1.0000 |
| train_c2_equals_event_c2.all | 0.7986 | 1.0000 | 1.0000 |
| train_c2_equals_event_c2.singleton_events | 0.8185 | 1.0000 | 1.0000 |
| train_c2_equals_event_c2.multi_article_events | 0.6029 | 1.0000 | 1.0000 |
| validation_inheritance.inherited_train_events | 30 | 30 | 30 |
| validation_inheritance.events_with_reused_and_new_articles | 0 | 0 | 0 |
| validation_inheritance.reused_and_new_same_c2_event_ratio | - | - | - |
| validation_inheritance.validation_only_c2_equals_event_c2 | 1.0000 | 1.0000 | 1.0000 |
| prefix2_c1c2.num_groups | 1129 | 944 | 1149 |
| prefix2_c1c2.entropy_bits | 8.9619 | 8.6628 | 9.0063 |
| prefix2_c1c2.max_group_size | 128 | 235 | 141 |
| prefix2_c1c2.median_group_size | 4.0000 | 4.0000 | 4.0000 |
| prefix2_c1c2.top10_group_share | 0.0869 | 0.1131 | 0.0878 |
| prefix2_c1c2.singleton_group_ratio | 0.2808 | 0.2606 | 0.2768 |
| prefix3_c1c2c3.num_groups | 10952 | 10761 | 10968 |
| prefix3_c1c2c3.entropy_bits | 13.2898 | 13.2517 | 13.2905 |
| prefix3_c1c2c3.max_group_size | 17 | 16 | 18 |
| prefix3_c1c2c3.median_group_size | 1.0000 | 1.0000 | 1.0000 |
| prefix3_c1c2c3.top10_group_share | 0.0083 | 0.0084 | 0.0085 |
| prefix3_c1c2c3.singleton_group_ratio | 0.8878 | 0.8764 | 0.8900 |
| train_event_size.max | 39 | 39 | 39 |
| train_event_size.p99 | 2.0000 | 2.0000 | 2.0000 |
| train_event_size.mean | 1.0648 | 1.0648 | 1.0648 |
| train_event_size.top5 | [39, 24, 24, 20, 15] | [39, 24, 24, 20, 15] | [39, 24, 24, 20, 15] |

### A 대비 변경

| metric | B | B-r1 |
|---|---|---|
| train.articles | 9738 | 9738 |
| train.c1_changed_count | 0 | 0 |
| train.c1_changed_ratio | 0.0000 | 0.0000 |
| train.c2_changed_count | 1961 | 338 |
| train.c2_changed_ratio | 0.2014 | 0.0347 |
| train.c3_changed_count | 1013 | 161 |
| train.c3_changed_ratio | 0.1040 | 0.0165 |
| train.c4_changed_count | 604 | 177 |
| train.c4_changed_ratio | 0.0620 | 0.0182 |
| train.singleton_event_articles | 8839 | 8839 |
| train.singleton_event_c2_changed_count | 1604 | 0 |
| train.singleton_event_c2_changed_ratio | 0.1815 | 0.0000 |
| train.multi_article_event_articles | 899 | 899 |
| train.multi_article_event_c2_changed_count | 357 | 338 |
| train.multi_article_event_c2_changed_ratio | 0.3971 | 0.3760 |
| validation.articles | 3122 | 3122 |
| validation.c1_changed_count | 0 | 0 |
| validation.c1_changed_ratio | 0.0000 | 0.0000 |
| validation.c2_changed_count | 0 | 819 |
| validation.c2_changed_ratio | 0.0000 | 0.2623 |
| validation.c3_changed_count | 0 | 627 |
| validation.c3_changed_ratio | 0.0000 | 0.2008 |
| validation.c4_changed_count | 279 | 292 |
| validation.c4_changed_ratio | 0.0894 | 0.0935 |
| all.articles | 12860 | 12860 |
| all.c1_changed_count | 0 | 0 |
| all.c1_changed_ratio | 0.0000 | 0.0000 |
| all.c2_changed_count | 1961 | 1157 |
| all.c2_changed_ratio | 0.1525 | 0.0900 |
| all.c3_changed_count | 1013 | 788 |
| all.c3_changed_ratio | 0.0788 | 0.0613 |
| all.c4_changed_count | 883 | 469 |
| all.c4_changed_ratio | 0.0687 | 0.0365 |

### reconstruction (별도 지표, 저장된 c1,c2,c3로 decode)

| metric | A | B | B-r1 |
|---|---|---|---|
| train.reconstruction_loss | 0.1365 | 0.1394 | 0.1372 |
| train.r2_norm_mean | 0.3954 | 0.4003 | 0.3968 |
| train.final_residual_norm_mean | 0.3407 | 0.3468 | 0.3423 |
| validation.reconstruction_loss | 0.1544 | 0.1544 | 0.1527 |
| validation.r2_norm_mean | 0.4018 | 0.4018 | 0.3974 |
| validation.final_residual_norm_mean | 0.3665 | 0.3665 | 0.3626 |

