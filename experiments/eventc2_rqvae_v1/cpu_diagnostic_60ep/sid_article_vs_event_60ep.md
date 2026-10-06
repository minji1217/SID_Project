| variant | role |
|---|---|
| A-article-60ep | - |
| event-60ep | - |

### SID 지표

| metric | A-article-60ep | event-60ep |
|---|---|---|
| counts.train_articles | 9738 | 9738 |
| counts.validation_articles | 3122 | 3122 |
| counts.validation_reused_train_articles | 0 | 0 |
| counts.validation_only_articles | 3122 | 3122 |
| counts.unique_articles | 12860 | 12860 |
| counts.train_events | 9145 | 9145 |
| counts.train_singleton_events | 8839 | 8839 |
| unique_codes.c1 | 25 | 25 |
| unique_codes.c2 | 79 | 53 |
| unique_codes.c3 | 505 | 83 |
| unique_codes.c12 | 819 | 487 |
| unique_codes.c123 | 10631 | 5217 |
| unique_codes.c1234 | 12860 | 12860 |
| collision.c123_collision_article_ratio | 0.2820 | 0.7558 |
| collision.c123_collision_rate | 0.1733 | 0.5943 |
| collision.max_c4 | 17 | 104 |
| same_event_c2.train.multi_article_events | 306 | 306 |
| same_event_c2.train.all_same_c2_event_ratio | 0.2876 | 1.0000 |
| same_event_c2.train.dominant_c2_ratio_mean | 0.6475 | 1.0000 |
| same_event_c2.validation.multi_article_events | 72 | 72 |
| same_event_c2.validation.all_same_c2_event_ratio | 1.0000 | 1.0000 |
| same_event_c2.validation.dominant_c2_ratio_mean | 1.0000 | 1.0000 |
| same_event_c2.all_unique_articles.multi_article_events | 402 | 402 |
| same_event_c2.all_unique_articles.all_same_c2_event_ratio | 0.4353 | 1.0000 |
| same_event_c2.all_unique_articles.dominant_c2_ratio_mean | 0.7215 | 1.0000 |
| train_c2_equals_event_c2.all | 0.6428 | 1.0000 |
| train_c2_equals_event_c2.singleton_events | 0.6543 | 1.0000 |
| train_c2_equals_event_c2.multi_article_events | 0.5306 | 1.0000 |
| validation_inheritance.inherited_train_events | 30 | 30 |
| validation_inheritance.events_with_reused_and_new_articles | 0 | 0 |
| validation_inheritance.reused_and_new_same_c2_event_ratio | - | - |
| validation_inheritance.validation_only_c2_equals_event_c2 | 1.0000 | 1.0000 |
| prefix2_c1c2.num_groups | 819 | 487 |
| prefix2_c1c2.entropy_bits | 8.4773 | 7.3180 |
| prefix2_c1c2.max_group_size | 294 | 346 |
| prefix2_c1c2.median_group_size | 5.0000 | 6.0000 |
| prefix2_c1c2.top10_group_share | 0.1113 | 0.2257 |
| prefix2_c1c2.singleton_group_ratio | 0.2393 | 0.2587 |
| prefix3_c1c2c3.num_groups | 10631 | 5217 |
| prefix3_c1c2c3.entropy_bits | 13.2169 | 11.4545 |
| prefix3_c1c2c3.max_group_size | 18 | 105 |
| prefix3_c1c2c3.median_group_size | 1.0000 | 1.0000 |
| prefix3_c1c2c3.top10_group_share | 0.0112 | 0.0558 |
| prefix3_c1c2c3.singleton_group_ratio | 0.8686 | 0.6021 |
| train_event_size.max | 39 | 39 |
| train_event_size.p99 | 2.0000 | 2.0000 |
| train_event_size.mean | 1.0648 | 1.0648 |
| train_event_size.top5 | [39, 24, 24, 20, 15] | [39, 24, 24, 20, 15] |

### A-article-60ep 대비 변경

| metric | event-60ep |
|---|---|
| train.articles | 9738 |
| train.c1_changed_count | 0 |
| train.c1_changed_ratio | 0.0000 |
| train.c2_changed_count | 9587 |
| train.c2_changed_ratio | 0.9845 |
| train.c3_changed_count | 9707 |
| train.c3_changed_ratio | 0.9968 |
| train.c4_changed_count | 5455 |
| train.c4_changed_ratio | 0.5602 |
| train.singleton_event_articles | 8839 |
| train.singleton_event_c2_changed_count | 8693 |
| train.singleton_event_c2_changed_ratio | 0.9835 |
| train.multi_article_event_articles | 899 |
| train.multi_article_event_c2_changed_count | 894 |
| train.multi_article_event_c2_changed_ratio | 0.9944 |
| validation.articles | 3122 |
| validation.c1_changed_count | 0 |
| validation.c1_changed_ratio | 0.0000 |
| validation.c2_changed_count | 3037 |
| validation.c2_changed_ratio | 0.9728 |
| validation.c3_changed_count | 3115 |
| validation.c3_changed_ratio | 0.9978 |
| validation.c4_changed_count | 2280 |
| validation.c4_changed_ratio | 0.7303 |
| all.articles | 12860 |
| all.c1_changed_count | 0 |
| all.c1_changed_ratio | 0.0000 |
| all.c2_changed_count | 12624 |
| all.c2_changed_ratio | 0.9816 |
| all.c3_changed_count | 12822 |
| all.c3_changed_ratio | 0.9970 |
| all.c4_changed_count | 7735 |
| all.c4_changed_ratio | 0.6015 |

### reconstruction (별도 지표, 저장된 c1,c2,c3로 decode)

| metric | A-article-60ep | event-60ep |
|---|---|---|
| train.reconstruction_loss | 0.4156 | 0.1604 |
| train.r2_norm_mean | 0.3611 | 0.3165 |
| train.final_residual_norm_mean | 0.3636 | 0.2203 |
| validation.reconstruction_loss | 0.4200 | 0.1641 |
| validation.r2_norm_mean | 0.3522 | 0.3133 |
| validation.final_residual_norm_mean | 0.3555 | 0.2205 |

