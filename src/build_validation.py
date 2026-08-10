


# STEP 7-5. validation에서 실제 사용하는 기사 ID 수집
# validation history 기사 
# stable dedup 후 단일 클릭 behavior의 current 기사
# stable dedup 후 단일 클릭 behavior의 target 기사
# 를 하나의 unique article 집합으로 수집하기 위해 
# 이전에 우리가 구했던 train_used_article_ids처럼 

def _collect_validation_used_article_ids(valid_article_ids:set[int],)-> dict[str, Any]:
    """
    validation에서 실제 사용하는 article_id 수집
    train의 collect_train_used_article_ids()와 동일한 방법
    behavior는 stable dedup 후 클릭 target이 정확히 1개인 usable behavior만 사용
    """

    validation_history = pl.read_parquet(
        VALIDATION_HISTORY_PATH
    ).select([
        "user_id", "article_id_fixed",
    ])

    validation_behaviors = pl.read_parquet(
        VALIDATION_BEHAVIORS_PATH
    ).select([
        "impression_id","article_id","article_ids_clicked",
    ])

    raw_used_article_ids: set[int]= set()

    # STEP 7-5-1. validation history 기사 수집
    # 사용자의 초기 history에 등장한 모든 기사 id 수집
    for row in validation_history.iter_rows(named=True):
        article_ids=(row["article_id_fixed"])

        if article_ids is None: continue 

        for article_id in article_ids: 
            if article_id is None: continue 

            raw_used_article_ids.add(
                int(article_id)
            )
    usable_behavior_row_count = 0 

    # STEP 7-5-2. Validation usable behavior 기사 수집 
    # clicked list를 stable dedup한 뒤 target이 하나인 행만 사용
    for row in validation_behaviors.iter_rows(named=True):
        clicked_article_ids = row["article_ids_clicked"]

        # stable dedup : 처음 등장한 순서는 유지하며 중복 제거
        deduplicated_clicked_ids: list[int] = []
        seen_clicked_ids: set[int] = set()

        if clicked_article_ids is not None: 
            for article_id in clicked_article_ids:
                if article_id is None:
                    continue 

                article_id = int(article_id)
                if article_id in seen_clicked_ids: continue 
                seen_clicked_ids.add(article_id) # 지금까지 본 것들 
                deduplicated_clicked_ids.append(article_id) # 결과 리스트 

        # baseline : stable dedup 후 클릭 기사가 정확히 1개인 행만
        if len(deduplicated_clicked_ids) != 1: continue 
        usable_behavior_row_count += 1

        # target 기사 
        target_article_id = deduplicated_clicked_ids[0]

        raw_used_article_ids.add(target_article_id)

        # behavior의 current article
        current_article_id = row["article_id"]

        if current_article_id is not None:
            raw_used_article_ids.add(int(current_article_id))

    # STEP 7-5-3. 최종 valid article과 교집합 
    # build_valid_articles()에서 제외된 기사는 사건 생성에서도 제외해야함
    valid_used_article_ids = (raw_used_article_ids & valid_article_ids)

    excluded_article_ids = sorted(raw_used_article_ids - valid_article_ids)

    return {
        "used_article_ids": (
            valid_used_article_ids
        ),
        "raw_used_article_count": (
            len(raw_used_article_ids)
        ),
        "valid_used_article_count": (
            len(valid_used_article_ids)
        ),
        "excluded_article_count": (
            len(excluded_article_ids)
        ),
        "excluded_article_examples": (
            excluded_article_ids[:10]
        ),
        "usable_behavior_row_count": (
            usable_behavior_row_count
        ),
    }
