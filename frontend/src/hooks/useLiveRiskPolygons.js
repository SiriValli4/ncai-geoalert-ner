/**
 * Keeps a district FeatureCollection's risk_score / risk_level in sync with
 * live predictions from POST /risk/predict.
 *
 * A feature takes part only if `properties.prediction_input` is set (the
 * request body for that district). Everything else passes through untouched.
 *
 * Failure behaviour matters here -- this drives an emergency dashboard:
 *   * Backend unreachable on first load -> the static scores already in the
 *     FeatureCollection are shown, and status is 'offline'.
 *   * Backend drops out after a successful fetch -> each district keeps its
 *     last live prediction (never silently reverts to sample data), and
 *     status is 'offline' with `lastUpdated` telling the user how stale it is.
 *
 * Pass a stable `basePolygons` reference (module constant or useMemo):
 * a new object every render restarts the fetch cycle.
 */

import { useEffect, useRef, useState } from 'react'
import { predictRisk } from '../api/riskApi'
import { riskBandForScore } from '../riskBands'

const SEVERITY_TO_RISK_LEVEL = { LOW: 'Low', MEDIUM: 'Medium', HIGH: 'High' }

function applyPrediction(feature, prediction) {
  const score = Math.round(prediction.landslide_susceptibility_index)
  return {
    ...feature,
    properties: {
      ...feature.properties,
      risk_score: score,
      // Band from the score, not the model's argmax class: the score bands are
      // what SMS dispatch and the priority queue use, so the map must agree
      // with them. The two can differ near band edges (e.g. class HIGH at
      // score 65), so the model's class is kept for the popup.
      risk_level: riskBandForScore(score),
      risk_source: 'model',
      prediction: {
        model_severity: SEVERITY_TO_RISK_LEVEL[prediction.severity] || null,
        top_factors: prediction.top_factors || [],
        estimated_inputs: prediction.estimated_inputs || [],
        model_type: prediction.model_type,
      },
    },
  }
}

function markSample(feature) {
  return { ...feature, properties: { ...feature.properties, risk_source: 'sample' } }
}

/**
 * @returns {{
 *   polygons: object,
 *   status: 'disabled'|'loading'|'live'|'partial'|'offline',
 *   lastUpdated: Date|null,
 *   error: string|null,
 *   refresh: () => void,
 * }}
 */
export default function useLiveRiskPolygons(basePolygons, { enabled = true, refreshMs = 5 * 60_000 } = {}) {
  const [polygons, setPolygons] = useState(basePolygons)
  const [status, setStatus] = useState(enabled ? 'loading' : 'disabled')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)
  const [refreshToken, setRefreshToken] = useState(0)

  // Last successful prediction per feature index, survives failed refreshes.
  const lastLiveRef = useRef(new Map())

  // New base data invalidates cached predictions (different districts/inputs).
  useEffect(() => {
    lastLiveRef.current = new Map()
    setPolygons(basePolygons)
  }, [basePolygons])

  useEffect(() => {
    if (!enabled) {
      setStatus('disabled')
      return undefined
    }

    let controller = null

    async function fetchAll() {
      controller?.abort()
      controller = new AbortController()
      const { signal } = controller

      const features = basePolygons.features
      const targets = features
        .map((f, i) => [f, i])
        .filter(([f]) => f.properties?.prediction_input)

      if (targets.length === 0) {
        setStatus('disabled')
        return
      }

      // No batch endpoint exists; districts are few, so parallel single calls are fine.
      const results = await Promise.allSettled(
        targets.map(([f]) => predictRisk(f.properties.prediction_input, { signal })),
      )
      if (signal.aborted) return

      const byIndex = new Map()
      let ok = 0
      let firstError = null
      results.forEach((r, k) => {
        const idx = targets[k][1]
        if (r.status === 'fulfilled') {
          ok += 1
          byIndex.set(idx, r.value)
          lastLiveRef.current.set(idx, r.value)
        } else {
          firstError = firstError || r.reason
          const cached = lastLiveRef.current.get(idx)
          if (cached) byIndex.set(idx, cached)
        }
      })

      setPolygons({
        ...basePolygons,
        features: features.map((f, i) => {
          if (!f.properties?.prediction_input) return f
          const pred = byIndex.get(i)
          return pred ? applyPrediction(f, pred) : markSample(f)
        }),
      })

      if (ok === targets.length) {
        setStatus('live')
        setError(null)
      } else {
        setStatus(ok > 0 ? 'partial' : 'offline')
        setError(firstError?.message || 'Risk API unavailable')
      }
      if (ok > 0) setLastUpdated(new Date())
    }

    fetchAll()
    const interval = refreshMs > 0 ? setInterval(fetchAll, refreshMs) : null

    return () => {
      controller?.abort()
      if (interval) clearInterval(interval)
    }
  }, [basePolygons, enabled, refreshMs, refreshToken])

  return {
    polygons,
    status,
    lastUpdated,
    error,
    refresh: () => setRefreshToken((t) => t + 1),
  }
}
