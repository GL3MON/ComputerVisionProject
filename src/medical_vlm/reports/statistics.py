"""Paired binary-outcome comparisons from saved predictions."""

import math


def exact_mcnemar(wins, losses):
    discordant = wins + losses
    if not discordant:
        return 1.0
    return min(
        1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant
    )


def paired(left, right, key, indices):
    wins = sum(not left[i][key] and right[i][key] for i in indices)
    losses = sum(left[i][key] and not right[i][key] for i in indices)
    both_correct = sum(left[i][key] and right[i][key] for i in indices)
    both_wrong = sum(not left[i][key] and not right[i][key] for i in indices)
    assert wins + losses + both_correct + both_wrong == len(indices)
    assert wins - losses == sum(right[i][key] for i in indices) - sum(left[i][key] for i in indices)
    return {
        "examples": len(indices),
        "current_only_correct": wins,
        "previous_only_correct": losses,
        "both_correct": both_correct,
        "both_incorrect": both_wrong,
        "net_current_correct_change": wins - losses,
        "difference_percentage_points": (wins - losses) / len(indices) * 100,
        "exact_mcnemar_pvalue": exact_mcnemar(wins, losses),
    }
