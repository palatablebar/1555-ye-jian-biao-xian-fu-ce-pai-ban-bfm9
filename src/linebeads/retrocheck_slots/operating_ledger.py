from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from .dry_signing import MAX_AGE_GAP, DRY_CEIL, WET_FLOOR, sign_dry_comparisons


@dataclass(frozen=True, slots=True)
class SurfacePair:
    trail: str
    age: int
    light: float
    weather: str
    unresolved: bool


def read_luminance(sample):
    table = sample['measurements']
    wet = table['wet_pct']
    road = table['surface_c']
    environmental = '潮湿' if wet>=60 else '干面' if wet<=20 else '过渡'
    corrected = table['retro_cd'] - .9*(road-20)
    unsafe_to_compare = road<5 or wet>=75
    return SurfacePair(sample['key'],int(table['age_days']),corrected,environmental,unsafe_to_compare)


def usable_reference(source, target):
    return [item for item in source if item.trail!=target.trail and abs(item.age-target.age)<=70]


def operating_ledger(rows, **dry_config):
    routes = defaultdict(list)
    survey = []
    for location in rows:
        if location['unmeasured']:
            survey.append({'业务安排':'补抄路况与车流票','关联单据':location['key'],'可用于坐标放样':False})
            continue
        routes[location['zone']].append((read_luminance(location),location))
    for area, book in sorted(routes.items()):
        all_values = [point for point,_ in book]
        dry_index = [point for point,ticket in book if point.weather=='干面' and not point.unresolved
                     and ticket['state'] in {'已分派','观察'}]
        lane_factor = 0
        for point,ticket in sorted(book,key=lambda entry:(-entry[1]['measurements']['traffic_k'],entry[0].trail)):
            beads = ticket['measurements']['beads_g']
            traffic = ticket['measurements']['traffic_k']
            dry = usable_reference(dry_index,point)
            spread = point.light-median(item.light for item in dry) if dry else None
            reserve = 0
            if point.unresolved:
                course = '等干面再测，保留湿读数'
            elif point.weather=='潮湿' and not dry:
                course = '找相近龄期干测点'
            elif beads<210 or (spread is not None and spread<-78):
                course = '见证复测玻璃珠投放段'
                reserve = 2 + int(traffic>=40)
            elif ticket['estimate']>=25:
                course = '安排同路段三次读数'
                reserve = 3
            else:
                continue
            lane_factor += reserve
            survey.append({'业务安排':course,'关联单据':point.trail,'标线路段':area,'路面':point.weather,
                           '干面对照差':None if spread is None else round(spread,1),
                           '可对照测点':len(dry),'复测建议次数':reserve})
        survey.append({'业务安排':'夜间复测安排','标线路段':area,'预计复测读数':lane_factor,
                       '潮湿暂留测点':sum(item.unresolved for item in all_values),
                       '同组待配干面':sum(item.weather=='潮湿' and not usable_reference(dry_index,item) for item in all_values)})
    if dry_config.pop('enrich_dry', True):
        _attach_dry_signoffs(survey, rows, dry_config)
    return survey


def _row_number(row, name, default=None):
    measurements = row.get('measurements', {}) if isinstance(row, dict) else {}
    value = measurements.get(name, row.get(name, default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _attach_dry_signoffs(survey, rows, config):
    """Add exclusive pairing fields to dictionary-style unit/workbench rows."""
    row_list = [row for row in rows if isinstance(row, dict) and 'measurements' in row]
    wet_floor = float(config.pop('wet_floor', WET_FLOOR))
    dry_ceiling = float(config.pop('dry_ceiling', DRY_CEIL))
    max_age = float(config.pop('max_age', MAX_AGE_GAP))
    score_dry = config.pop('score_dry', None)
    dry_rows = [row for row in row_list if not row.get('unmeasured')
                and _row_number(row, 'wet_pct', 10**9) <= dry_ceiling]
    by_zone = defaultdict(list)
    for dry in dry_rows:
        by_zone[dry.get('zone', dry.get('cohort', '_默认区'))].append(dry)

    cases = []
    for row in row_list:
        wet = _row_number(row, 'wet_pct')
        if wet is None or wet < wet_floor:
            continue
        zone = row.get('zone', row.get('cohort', '_默认区'))
        age = _row_number(row, 'age_days')
        options = []
        explicit = row.get('dry_estimates') or row.get('dry_scores') or {}
        for ref in by_zone.get(zone, []):
            ref_key = ref.get('key')
            if ref_key == row.get('key'):
                continue
            ref_age = _row_number(ref, 'age_days')
            if age is None or ref_age is None or abs(age-ref_age) > max_age:
                continue
            if callable(score_dry):
                dry_estimate = score_dry(row, ref)
            else:
                dry_estimate = explicit.get(ref_key, ref.get('dry_estimate', ref.get('estimate')))
            options.append({'witness': ref_key, 'age_gap': abs(age-ref_age),
                            'dry_estimate': dry_estimate})
        cases.append({
            'key': row.get('key'), 'zone': zone,
            'cost': row.get('cost', 1), 'risk': row.get('wet_estimate', row.get('estimate', 0)),
            'state': row.get('state', ''), 'reason': row.get('reason', ''),
            'options': options,
        })

    plan = sign_dry_comparisons(cases, **config)
    by_ticket = {item['key']: item for item in plan}
    positions = {}
    for index, item in enumerate(survey):
        if '关联单据' in item:
            positions.setdefault(item['关联单据'], index)
    for key, dry in by_ticket.items():
        fields = {k: v for k, v in dict(dry).items() if k != 'candidate_pairs'}
        if key in positions:
            target = survey[positions[key]]
            preserved_arrangement = target.get('业务安排')
            target.update(fields)
            if preserved_arrangement:
                target['业务安排'] = preserved_arrangement
        else:
            survey.append(fields)


def __getattr__(name):
    from .dry_signing import universal_attribute
    try:
        return universal_attribute(name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
