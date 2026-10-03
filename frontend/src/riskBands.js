/**
 * Risk score (0-100) -> band. Must match the rest of the system:
 * services/alerts/dispatchAlertsRoute.js (HIGH_RISK_THRESHOLD = 70,
 * MEDIUM_RISK_THRESHOLD = 40) and the Domain constants in CLAUDE.md.
 */
export const HIGH_RISK_THRESHOLD = 70
export const MEDIUM_RISK_THRESHOLD = 40

export function riskBandForScore(score) {
  if (score >= HIGH_RISK_THRESHOLD) return 'High'
  if (score >= MEDIUM_RISK_THRESHOLD) return 'Medium'
  return 'Low'
}
