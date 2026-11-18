"""CPU-only byte-budget allocation for the expert cache's size classes."""


def split_slots(budget, layer_counts, record_bytes, fixed, minimum=None):
    """Share bytes by layer count, honoring fixed pools and minimum slot counts."""
    minimum = minimum or [0] * len(layer_counts)
    slots = [f if n == 0 else 0 for n, f in zip(layer_counts, fixed)]
    remaining = budget - sum(s * r for s, r in zip(slots, record_bytes))
    active = {c for c, n in enumerate(layer_counts) if n}
    required = sum(minimum[c] * record_bytes[c] for c in active)
    if remaining < required:
        raise ValueError(f"expert cache budget {budget} bytes cannot fit minimum pools "
                         f"({budget - remaining + required} bytes)")
    while active:
        per = sum(layer_counts[c] * record_bytes[c] for c in active)
        proposed = {c: remaining * layer_counts[c] // per for c in active}
        constrained = {c for c in active if proposed[c] < minimum[c]}
        if not constrained:
            for c in active:
                slots[c] = proposed[c]
            break
        for c in constrained:
            slots[c] = minimum[c]
            remaining -= slots[c] * record_bytes[c]
        active -= constrained
    return slots
