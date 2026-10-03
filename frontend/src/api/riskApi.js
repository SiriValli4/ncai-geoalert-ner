/**
 * Client for the GeoAlert FastAPI backend (app/api/v1/endpoints/risk.py).
 *
 * Base URL comes from REACT_APP_API_BASE_URL (CRA bakes it in at build time).
 * The backend's CORS_ORIGINS must include this app's origin -- its default,
 * http://localhost:3000, matches `npm start`.
 */

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1'
).replace(/\/+$/, '')

const DEFAULT_TIMEOUT_MS = 10_000

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * POST /risk/predict
 *
 * @param {{latitude:number, longitude:number, rainfall_24h_mm:number,
 *          soil_moisture_percent:number, slope_degree:number}} input
 * @param {{signal?: AbortSignal, timeoutMs?: number}} [options]
 * @returns {Promise<{landslide_susceptibility_index:number,
 *                    severity:'LOW'|'MEDIUM'|'HIGH',
 *                    top_factors:Array<{feature:string, explanation:string}>,
 *                    estimated_inputs:string[], model_type:string}>}
 */
export async function predictRisk(input, { signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  // Combine the caller's abort signal (unmount/refresh) with a timeout, so a
  // hung backend can't leave the dashboard "Connecting..." forever.
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort)

  try {
    const res = await fetch(`${API_BASE_URL}/risk/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
      signal: controller.signal,
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      const detail = typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`
      throw new ApiError(detail, res.status)
    }
    return await res.json()
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}
