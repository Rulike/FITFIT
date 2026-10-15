def detect_conflicts(records):
    fields={'event_time':'event_time_conflict','location':'location_conflict','event_status':'status_conflict','event_object':'object_conflict','impact_direction':'impact_direction_conflict'}
    out=[]
    for f,kind in fields.items():
        vals={str(r.get(f)) for r in records if r.get(f) not in (None,'',[])}
        if len(vals)>1: out.append({'type':kind,'values':sorted(vals)})
    return out
