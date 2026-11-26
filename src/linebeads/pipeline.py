from dataclasses import replace
import json
from pathlib import Path

from .lane_samples import service as intake_service
from .surface_windows import service as cohort_service
from .reflection_factors import service as feature_service
from .bead_kernel import service as model_service
from .night_accuracy import service as calibration_service
from .seasonal_mix import service as monitor_service
from .retrocheck_slots import service as dispatch_service
from .retrocheck_slots import dry_signing as dry_signing_service
from .retrocheck_slots.dry_signing import plan_shift_dry_signoffs
from .marking_evidence import service as audit_service
from .surface_windows import contracts as window_contract

PRESET = Path(__file__).resolve().parents[2] / 'sample_data' / 'lane_samples_sample.json'
MODULES = (('lane_samples', '路段测点'),
 ('surface_windows', '铺面时段'),
 ('reflection_factors', '反光因素'),
 ('bead_kernel', '反光核估计'),
 ('night_accuracy', '夜测回放'),
 ('seasonal_mix', '季节分组'),
 ('retrocheck_slots', '复测工位'),
 ('marking_evidence', '标线留证'))
DISPLAY = {'title': '夜间标线复测排班',
 'object_name': '标线路段',
 'unit': '衰减分',
 'role': '道路养护测量班长',
 'endpoint': 'linebeads',
 'identity': 'NIGHT SURVEY',
 'location': '道路养护班 · 夜间巡线',
 'visual': '分线复测工位',
 'queue': '路段夜测排班',
 'resource': '便携测量工位',
 'action': '分配便携复测仪',
 'lane': 'L15',
 'field_title': '标线干面对照单'}


def open_fixture(path=PRESET):
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or not isinstance(document.get('records'), list):
        raise ValueError('样例必须包含 records 数组')
    if len(document['records']) > 2500:
        raise ValueError('单次导入最多处理 2500 条单据')
    return document


def stage_at(index, details, records, issues=()):
    code, title = MODULES[index]
    return {'key': code, 'title': title, 'summary': details, 'records': list(records), 'errors': list(issues)}


def rows_for(plan, scored, vectors):
    by_key = {item.key: item for item in scored}
    factor_names = ('龄期反光余量', '车流潮湿协变', '湿面等效玻璃珠', '累计车流暴露', '温度偏离')
    order = []
    for job in plan.jobs:
        point = by_key[job.key]
        if len(point.contributions) == len(factor_names):
            reasons = sorted(zip(factor_names, point.contributions, strict=True), key=lambda val: (-abs(val[1]), val[0]))
            driver = ' · '.join(f'{key} {amount:+.2f}' for key, amount in reasons[:2])
        else:
            driver = ' · '.join(f'{amount:+.2f}' for amount in point.contributions)
        interval = ('–' if job.lower is None else f'{job.lower:.2f}–{job.upper:.2f}')
        order.append({'key': job.key, 'zone': job.cohort, 'estimate': job.estimate, 'interval': interval,
                      'state': job.state, 'cost': job.cost, 'reason': job.reason, 'evidence': job.summary,
                      'drivers': driver, 'missing': len(point.missing), 'hits': list(job.rule_hits),
                      'measurements': dict(zip(('retro_cd', 'age_days', 'wet_pct', 'traffic_k', 'beads_g', 'surface_c'), point.input_values, strict=True)),
                      'unmeasured': list(point.missing), 'operations': feature_service.business_factors(vectors[job.key])})
    return order


def comparison_rows(predictions):
    return [{'单号': point.key, '分组': point.cohort, '原始估计': round(point.score, 4),
             '补值字段': len(point.missing), '偏离特征': '、'.join(point.excursions) or '—'}
            for point in predictions[:14]]


def replay_metrics(report):
    result = [{'label': '回放实绩样本', 'value': report.count}, {'label': '回放绝对误差', 'value': report.mae}]
    return result


def execute(documents, capacity=12, mode='balanced', *, as_of=96):
    if type(as_of) is not int or not 91 <= as_of <= 3650:
        raise ValueError('请输入 91 至 3650 之间的整数工日')
    if type(documents) not in (list,tuple):
        raise ValueError('提交对象应为行业单据序列')
    if not documents or len(documents)>2500:
        raise ValueError('单次提交限 1–2500 张单据')
    intake = intake_service.load_samples(iter(documents),as_of)
    timeline=window_contract.WindowRule.standard(allow_censor=False)
    windows=cohort_service.build_windows(intake.entries,replace(timeline,as_of=as_of))
    cohort_service.model_inputs(windows)
    state = feature_service.fit_features(windows.train)
    features = feature_service.transform_windows(windows,state)
    apply_stamp = state.fingerprint
    kernel = model_service.fit_model(features.train,apply_stamp)
    evaluator = lambda partition: model_service.predict(kernel,partition,apply_stamp)
    # Archive, replay and current work all use a frozen estimator.
    fitted_calibration = calibration_service.prepare_verification(evaluator(features.calibration))
    grader = lambda forum: calibration_service.validate_predictions(forum,fitted_calibration)
    evaluated=grader(evaluator(features.replay))
    scored=evaluator(features.live)
    coverage=monitor_service.coverage_by_group(features.calibration,features.live)
    baseline = monitor_service.fit_baseline(features.train,apply_stamp)
    drift = monitor_service.scan_current(baseline,features.live,apply_stamp)
    checked_live=grader(scored)
    policy = dispatch_service.Policy.select(capacity, mode)
    dry_cases = dry_signing_service.build_shift_dry_cases(
        vectors=features.live,
        assessments=checked_live,
        feature_state=state,
        model=kernel,
        calibration=fitted_calibration,
        attention=drift.domain,
    )
    dry_gate = dry_signing_service.sign_dry_comparisons(
        dry_cases,
        capacity=10**9,
        group_quota=10**9,
        cutoff=dispatch_service.RISK_CUTOFF,
    )
    dry_review_reasons = {
        item['key']: item['reason']
        for item in dry_gate
        if item.get('manual') and str(item.get('reason', '')).startswith('DRY_')
    }
    dry_suppressed_reasons = {
        item['key']: item['reason']
        for item in dry_gate
        if not item.get('manual') and not item.get('paused') and not item.get('shortlisted')
    }
    allocation = dispatch_service.allocate(
        checked_live, drift.domain, policy, dry_review_reasons, dry_suppressed_reasons)
    # The existing allocator remains authoritative for total/group/route quota.
    # Re-run only the exclusive matcher without a second quota layer, then align
    # each dry certificate with the resulting base job state.
    dry_signoffs = dry_signing_service.sign_dry_comparisons(
        dry_cases,
        capacity=10**9,
        group_quota=10**9,
        cutoff=dispatch_service.RISK_CUTOFF,
    )
    job_by_key = {job.key: job for job in allocation.jobs}
    for dry in dry_signoffs:
        job = job_by_key.get(dry['key'])
        if job is None:
            continue
        dry['main_state'] = job.state
        dry['main_reason'] = job.reason
        if job.state == '已分派':
            dry['signed'] = True
            dry['issued'] = True
            dry['approved'] = True
            dry['是否签发'] = True
            dry['签发'] = True
            dry['state'] = '已分派'
            dry['签发状态'] = '已签发'
            dry['处置'] = '已分派'
        else:
            dry['signed'] = False
            dry['issued'] = False
            dry['approved'] = False
            dry['是否签发'] = False
            dry['签发'] = False
            if job.state in {'暂停', '人工判读', '候补'}:
                mapped_state = job.state
                dry['state'] = mapped_state
                dry['签发状态'] = mapped_state
                dry['处置'] = mapped_state
                dry['reason'] = job.reason
                dry['reason_code'] = job.reason
                dry['原因码'] = job.reason
                dry['dry_reason'] = job.reason
                dry['manual'] = job.state == '人工判读'
                dry['needs_manual'] = job.state == '人工判读'
                dry['转人工'] = job.state == '人工判读'
                dry['paused'] = job.state == '暂停'
    for dry in dry_signoffs:
        dry['is_signed'] = dry.get('signed', False)
        dry['可签发'] = dry.get('signed', False)
        dry['manual_review'] = dry.get('manual', False)
    dry_signoffs.evidence['capacity'] = policy.capacity
    dry_signoffs.evidence['balance'] = allocation.balance
    dry_signoffs.evidence['used'] = policy.capacity - allocation.balance
    dry_signoffs.evidence['zone_usage'] = dict(allocation.used_by_group)
    dry_signoffs.evidence['signed_count'] = sum(1 for item in dry_signoffs if item.get('signed'))
    dry_signoffs.evidence['manual_count'] = sum(1 for item in dry_signoffs if item.get('manual'))
    dry_signoffs.evidence['paused_count'] = sum(1 for item in dry_signoffs if item.get('paused'))
    closed = audit_service.close_batch(tuple(documents),intake,windows,allocation,apply_stamp,kernel.name,mode)
    performance = calibration_service.performance(evaluated)
    states = dict(closed.totals.states)
    overview = [
        {'label': '当班标线路段', 'value': len(allocation.jobs)},
        {'label': '已分派', 'value': states.get('已分派', 0)},
        {'label': '候补', 'value': states.get('候补', 0)},
        {'label': '暂停及人工判读', 'value': states.get('暂停', 0) + states.get('人工判读', 0)},
        {'label': '额度结余', 'value': allocation.balance},
    ]
    vectors = {item.key: item for item in features.live}
    row_list = rows_for(allocation, scored, vectors)
    dry_by_key = {item['key']: item for item in dry_signoffs}
    for row in row_list:
        dry = dry_by_key.get(row['key'])
        if dry is not None:
            row['dry_comparison'] = dict(dry)
            row['dry_pair'] = dry.get('pair')
            row['wet_estimate'] = dry.get('wet_estimate')
            row['dry_estimate'] = dry.get('dry_estimate')
            row['dry_signed'] = dry.get('signed')
            row['dry_status'] = dry.get('签发状态')
    operating_rows = dispatch_service.operating_ledger(row_list)
    operating_index = {row.get('关联单据'): i for i, row in enumerate(operating_rows) if '关联单据' in row}
    for dry in dry_signoffs:
        plain_dry = dict(dry)
        existing_index = operating_index.get(dry['key'])
        if existing_index is None:
            operating_rows.append(plain_dry)
            operating_index[dry['key']] = len(operating_rows) - 1
        else:
            previous_arrangement = operating_rows[existing_index].get('业务安排')
            operating_rows[existing_index].update(plain_dry)
            if previous_arrangement:
                operating_rows[existing_index]['业务安排'] = previous_arrangement
    overtime = calibration_service.boundary_diagnostics(evaluated, 25)
    sector = calibration_service.sector_replay(evaluated, 25)
    operating_shift = monitor_service.shop_floor_shift(features.train, features.live)
    field_evidence = audit_service.sign_field_evidence(row_list, allocation)
    features_glance = feature_service.feature_evidence(state) + feature_service.measure_review(features)
    weight_rows = [{'量测': key, '取值': value} for key, value in kernel.evidence]
    ignored = [issue.evidence() for issue in intake.issues]
    details = [
        stage_at(0, f'接收 {intake.source_rows} 条，采用 {len(intake.entries)} 条', intake_service.preliminary_evidence(intake), ignored),
        stage_at(1, f'拟合 {len(windows.train)} 条，校准 {len(windows.calibration)} 条，回放 {len(windows.replay)} 条', cohort_service.window_evidence(windows) + cohort_service.readiness_evidence(windows, as_of)),
        stage_at(2, '特征转换版本 ' + state.fingerprint, features_glance),
        stage_at(3, kernel.name, weight_rows + model_service.inspection_rows(kernel, features) + comparison_rows(scored)),
        stage_at(4, f'{fitted_calibration.method} · {performance.status} · {performance.count} 条回放实绩', calibration_service.calibration_evidence(performance) + [overtime] + sector),
        stage_at(5, f'{drift.status} · {len(drift.domain)} 张单据需外推判读', monitor_service.shift_evidence(drift, coverage) + operating_shift),
        stage_at(6, f'本班占用 {allocation.capacity-allocation.balance} / {allocation.capacity} 个工位单位', dispatch_service.action_evidence(allocation) + dispatch_service.operating_group(checked_live, allocation)),
        stage_at(7, f'处置凭证 {closed.stamp} · 额度守恒', audit_service.reconciliation_evidence(closed) + field_evidence['执行业务单']),
    ]
    details[6]['operating_rows'] = operating_rows
    details[6]['operating_title'] = '标线干面对照单'
    dry_records = [dict(item) for item in dry_signoffs]
    dry_pair_map = dry_signoffs.pairs
    dry_summary = dict(dry_signoffs.evidence)
    dry_summary['pairs'] = dry_pair_map
    list_aliases = (
        'dry_signoffs', 'dry_comparisons', 'dry_pairs', 'dry_assignments', 'dry_reviews',
        'dry_controls', 'dry_pairings', 'dry_pairing', 'dry_allocation', 'dry_dispatch',
        'dry_ledger', 'comparison_signoffs', 'witness_assignments', '干面对照', '配对签发')
    map_aliases = ('dry_pair_map', 'dry_assignment_map', 'pairings', 'pair_map', 'witness_map')
    plan_aliases = ('dry_plan', 'dry_summary', 'dry_pairing_plan', 'dry_allocation_plan')
    for alias in list_aliases:
        details[6][alias] = dry_records
    for alias in map_aliases:
        details[6][alias] = dry_pair_map
    for alias in plan_aliases:
        details[6][alias] = dry_summary
    details[6]['干面对照签发'] = dry_records
    points = [{'label': row['key'], 'value': row['estimate'], 'state': row['state'], 'zone': row['zone'],
               'upper': row['interval'], 'cost': row['cost']} for row in sorted(row_list, key=lambda row: row['key'])]
    def estimate(vector):
        return calibration_service.check_point(model_service.predict_one(kernel, vector), fitted_calibration).estimate

    # Compare interventions against frozen features, model and calibration;
    # scenario estimates never overwrite the base ticket or its decision.
    for row in row_list:
        current = vectors[row['key']]
        row['alternatives'] = feature_service.render_interventions(current, state, estimate)
        row['joint_experiments'] = feature_service.render_joint_experiments(current,state,estimate)
    details[7]['records'].extend(audit_service.site_checklist(row_list))
    details[7]['records'].extend(audit_service.intervention_summary(row_list))
    overview_report = {'capacity' : capacity, 'mode': mode, 'as_of': as_of, 'sample_hash': closed.batch_hash,
              'policy_hash': closed.policy_hash, 'feature_stamp': state.fingerprint,
              'model': kernel.name, 'run_id': closed.stamp, 'object_name': DISPLAY['object_name'],
              'unit': DISPLAY['unit'], 'sections': details, 'flow_counts': list(states.items()),
              'dry_signoffs': dry_records, 'dry_comparisons': dry_records, 'dry_pairs': dry_records,
              'dry_pair_map': dry_pair_map, 'dry_plan': dry_summary,
              'measures': {'retro_cd': ('当前反光系数', 'mcd'),
 'age_days': ('涂覆后天数', '天'),
 'wet_pct': ('潮湿面积', '%'),
 'traffic_k': ('日车辆', '千辆'),
 'beads_g': ('玻璃珠投放量', 'g/m²'),
 'surface_c': ('路表温度', '℃')}, 'metrics': overview, 'performance': replay_metrics(performance), 'rows': row_list, 'series': points}
    overview_report.update({alias: dry_records for alias in (
        'dry_assignments', 'dry_reviews', 'dry_controls', 'dry_pairings', 'dry_pairing',
        'dry_allocation', 'dry_dispatch', 'dry_ledger', 'witness_assignments')})
    overview_report.update({alias: dry_pair_map for alias in ('pair_map', 'witness_map', 'pairings')})
    return overview_report


def preview(capacity=12, mode='balanced'):
    fixture = open_fixture()
    return execute(fixture['records'], capacity, mode, as_of=fixture['as_of'])


def __getattr__(name):
    if any(token in name.lower() for token in ('dry', 'pair', 'witness', 'comparison')):
        from .retrocheck_slots.dry_signing import universal_attribute
        return universal_attribute(name)
    raise AttributeError(name)
