"""Size whole-unit entries from cash available at the actual fill."""

from fractions import Fraction
from math import isfinite, ulp


def cash_rounding_tolerance(*amounts: float) -> float:
    """Allow only arithmetic rounding dust when comparing cash amounts."""
    # The relative cap prevents a few subnormal ULPs becoming a material
    # fraction of an exceptionally small cash balance.
    return min(4 * max(ulp(value) for value in amounts), min(abs(value) for value in amounts) * 1e-12)


def capital_quantity(
    cash: float, allocation_pct: float, price: float,
    point_value: float, fee_rate: float,
) -> int:
    """Return the largest affordable whole position, including entry fees.

    Decimal-string fractions avoid rounding an exact quotient down before
    converting it to whole units. Check the neighboring unit against the
    executor's float arithmetic so a few rounding bits cannot reject an
    otherwise fully funded entry.
    """
    if cash <= 0:
        return 0
    budget = cash * (allocation_pct / 100.0)
    unit_cost = price * point_value * (1.0 + fee_rate)
    if budget <= 0 or not isfinite(unit_cost) or unit_cost <= 0:
        return 0
    exact_budget = Fraction(str(cash)) * Fraction(str(allocation_pct)) / 100
    exact_unit_cost = (
        Fraction(str(price)) * Fraction(str(point_value))
        * (1 + Fraction(str(fee_rate)))
    )
    quantity = exact_budget // exact_unit_cost

    def affordable(candidate):
        notional = candidate * price * point_value
        cost = notional + notional * fee_rate
        return isfinite(cost) and cost <= budget + cash_rounding_tolerance(budget, cost)

    if quantity and not affordable(quantity):
        quantity -= 1
    if quantity < 2 ** 53 and affordable(quantity + 1):
        quantity += 1
    return quantity
