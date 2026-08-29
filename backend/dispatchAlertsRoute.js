'use strict';
 
const express = require('express');
const twilio = require('twilio');
 
// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------
 
// A risk_score at or above this is HIGH. Kept consistent with the risk
// banding used elsewhere in this system (dashboard, priority queue, etc.).
const HIGH_RISK_THRESHOLD = 70;
const MEDIUM_RISK_THRESHOLD = 40;
 
const SUPPORTED_LANGUAGES = ['en', 'as', 'bn', 'mni'];
const DEFAULT_LANGUAGE = 'en';
 
function riskBandForScore(score) {
  if (score >= HIGH_RISK_THRESHOLD) return 'HIGH';
  if (score >= MEDIUM_RISK_THRESHOLD) return 'MEDIUM';
  return 'LOW';
}
 
// --------------------------------------------------------------------------
// Emergency templates (see disclaimer above)
// --------------------------------------------------------------------------
 
function formatHighwaysList(affectedHighways) {
  if (!affectedHighways || affectedHighways.length === 0) return null
  return affectedHighways.join(', ')
}
 
function buildAlertTemplates({ districtName, riskScore, affectedHighways }) {
  const highways = formatHighwaysList(affectedHighways)
 
  const en = highways
    ? `EMERGENCY ALERT: ${districtName} district is at HIGH landslide risk (score ${riskScore}/100). Avoid travel on: ${highways}. Stay away from slopes and riverbanks. Follow local authorities' instructions immediately. - NER Landslide Early Warning System`
    : `EMERGENCY ALERT: ${districtName} district is at HIGH landslide risk (score ${riskScore}/100). Stay away from slopes and riverbanks. Follow local authorities' instructions immediately. - NER Landslide Early Warning System`
 
  const as = highways
    ? `জৰুৰীকালীন সতৰ্কবাণী: ${districtName} জিলাত উচ্চ মাত্ৰাৰ ভূমিস্খলনৰ আশংকা আছে (স্কোৰ ${riskScore}/100)। ${highways} পথেৰে যাতায়াত নকৰিব। পাহাৰৰ চাল আৰু নদীৰ পাৰৰ পৰা আঁতৰি থাকক। স্থানীয় প্ৰশাসনৰ নিৰ্দেশনা তৎক্ষণাৎ মানি চলক। - NER ভূমিস্খলন সতৰ্কতা ব্যৱস্থা`
    : `জৰুৰীকালীন সতৰ্কবাণী: ${districtName} জিলাত উচ্চ মাত্ৰাৰ ভূমিস্খলনৰ আশংকা আছে (স্কোৰ ${riskScore}/100)। পাহাৰৰ চাল আৰু নদীৰ পাৰৰ পৰা আঁতৰি থাকক। স্থানীয় প্ৰশাসনৰ নিৰ্দেশনা তৎক্ষণাৎ মানি চলক। - NER ভূমিস্খলন সতৰ্কতা ব্যৱস্থা`
 
  const bn = highways
    ? `জরুরি সতর্কবার্তা: ${districtName} জেলায় উচ্চ মাত্রার ভূমিধসের আশঙ্কা রয়েছে (স্কোর ${riskScore}/100)। ${highways} রাস্তায় যাতায়াত এড়িয়ে চলুন। পাহাড়ি ঢাল ও নদীর তীর থেকে দূরে থাকুন। স্থানীয় প্রশাসনের নির্দেশ অবিলম্বে মেনে চলুন। - NER ভূমিধস সতর্কতা ব্যবস্থা`
    : `জরুরি সতর্কবার্তা: ${districtName} জেলায় উচ্চ মাত্রার ভূমিধসের আশঙ্কা রয়েছে (স্কোর ${riskScore}/100)। পাহাড়ি ঢাল ও নদীর তীর থেকে দূরে থাকুন। স্থানীয় প্রশাসনের নির্দেশ অবিলম্বে মেনে চলুন। - NER ভূমিধস সতর্কতা ব্যবস্থা`
 
  // Romanized Manipuri -- see the module-level disclaimer for why.
  const mni = highways
    ? `EMERGENCY ALERT: ${districtName} da landslide-gi high risk lei (score ${riskScore}/100). ${highways} lambida chatthoklaganu. Ching amasung turel mapanda leptaganu. Local admin-gi wayel adu ngaakna illu. - NER Landslide Warning System`
    : `EMERGENCY ALERT: ${districtName} da landslide-gi high risk lei (score ${riskScore}/100). Ching amasung turel mapanda leptaganu. Local admin-gi wayel adu ngaakna illu. - NER Landslide Warning System`
 
  return { en, as, bn, mni }
}
 
// --------------------------------------------------------------------------
// Registered recipients -- replace with a real subscriber database
// --------------------------------------------------------------------------
//
// This is sample/mock data using obviously-fake numbers so nothing here
// accidentally resembles a real subscriber. In production, swap
// `defaultRecipientProvider` for a query against your actual opt-in SMS
// registry (phone number + preferred language per district).
 
const SAMPLE_REGISTERED_RECIPIENTS = {
  'East Sikkim': [
    { phoneNumber: '+910000000001', preferredLanguage: 'en' },
    { phoneNumber: '+910000000002', preferredLanguage: 'as' },
  ],
  'North Sikkim': [{ phoneNumber: '+910000000003', preferredLanguage: 'en' }],
  'East Khasi Hills': [
    { phoneNumber: '+910000000004', preferredLanguage: 'bn' },
    { phoneNumber: '+910000000005', preferredLanguage: 'en' },
  ],
  'Dima Hasao': [{ phoneNumber: '+910000000006', preferredLanguage: 'as' }],
  Senapati: [{ phoneNumber: '+910000000007', preferredLanguage: 'mni' }],
}
 
function defaultRecipientProvider(districtName) {
  return SAMPLE_REGISTERED_RECIPIENTS[districtName] || []
}
 
// --------------------------------------------------------------------------
// Twilio client (real or mock)
// --------------------------------------------------------------------------
 
function readTwilioConfigFromEnv(env = process.env) {
  return {
    accountSid: env.TWILIO_ACCOUNT_SID,
    authToken: env.TWILIO_AUTH_TOKEN,
    fromNumber: env.TWILIO_FROM_NUMBER,
    messagingServiceSid: env.TWILIO_MESSAGING_SERVICE_SID,
  }
}
 
function hasTwilioCredentials(config) {
  return Boolean(config.accountSid && config.authToken && (config.fromNumber || config.messagingServiceSid))
}
 
/** Wraps either a real Twilio client or a mock so callers don't need to
 * care which one they got. `send()` always returns
 * `{ sid, status, mode }` or throws. */
function createSmsGateway(config = readTwilioConfigFromEnv()) {
  if (hasTwilioCredentials(config)) {
    const client = twilio(config.accountSid, config.authToken)
    return {
      mode: 'live',
      async send({ to, body }) {
        const message = await client.messages.create({
          to,
          body,
          ...(config.messagingServiceSid
            ? { messagingServiceSid: config.messagingServiceSid }
            : { from: config.fromNumber }),
        })
        return { sid: message.sid, status: message.status, mode: 'live' }
      },
    }
  }
 
  console.warn(
    '[dispatch-alerts] Twilio credentials not found (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / ' +
      'TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID) -- running in MOCK SMS mode. No real ' +
      'messages will be sent. Set those environment variables to send real SMS via Twilio.',
  )
  return {
    mode: 'mock',
    async send({ to, body }) {
      // Small artificial delay so this behaves like a real network call in
      // tests/demos rather than resolving suspiciously instantly.
      await new Promise((resolve) => setTimeout(resolve, 10))
      const sid = `MOCKSID_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      console.log(`[dispatch-alerts][MOCK] would send to ${to} (${body.length} chars): "${body}"`)
      return { sid, status: 'queued_mock', mode: 'mock' }
    },
  }
}
 
// --------------------------------------------------------------------------
// Request validation
// --------------------------------------------------------------------------
 
function validateRequestBody(body) {
  const errors = []
 
  if (!body || typeof body !== 'object') {
    return ['Request body must be a JSON object.']
  }
 
  if (typeof body.district_name !== 'string' || body.district_name.trim().length === 0) {
    errors.push('district_name is required and must be a non-empty string.')
  }
 
  if (typeof body.risk_score !== 'number' || Number.isNaN(body.risk_score)) {
    errors.push('risk_score is required and must be a number.')
  } else if (body.risk_score < 0 || body.risk_score > 100) {
    errors.push('risk_score must be between 0 and 100.')
  }
 
  if (body.affected_highways !== undefined) {
    const isStringArray =
      Array.isArray(body.affected_highways) && body.affected_highways.every((h) => typeof h === 'string')
    if (!isStringArray) {
      errors.push('affected_highways must be an array of strings when provided.')
    }
  }
 
  return errors
}
 
// --------------------------------------------------------------------------
// Router factory
// --------------------------------------------------------------------------
 
/**
 * @param {object} [options]
 * @param {(districtName: string) => Array<{phoneNumber: string, preferredLanguage: string}>} [options.recipientProvider]
 *   Override to plug in a real subscriber database. Defaults to the sample
 *   in-memory registry above.
 * @param {object} [options.twilioConfig] Override for testing; defaults to
 *   reading TWILIO_* from process.env.
 */
function createDispatchAlertsRouter(options = {}) {
  const recipientProvider = options.recipientProvider || defaultRecipientProvider
  const smsGateway = createSmsGateway(options.twilioConfig || readTwilioConfigFromEnv())
 
  const router = express.Router()
 
  router.post('/dispatch-alerts', async (req, res, next) => {
    try {
      const validationErrors = validateRequestBody(req.body)
      if (validationErrors.length > 0) {
        return res.status(400).json({ error: 'Invalid request', details: validationErrors })
      }
 
      const { district_name: districtName, risk_score: riskScore } = req.body
      const affectedHighways = req.body.affected_highways || []
 
      const riskBand = riskBandForScore(riskScore)
      if (riskBand !== 'HIGH') {
        return res.status(200).json({
          dispatched: false,
          reason: `risk_score ${riskScore} is ${riskBand}; alerts only dispatch at HIGH (>= ${HIGH_RISK_THRESHOLD}).`,
          districtName,
          riskScore,
          riskBand,
        })
      }
 
      const templates = buildAlertTemplates({ districtName, riskScore, affectedHighways })
      const recipients = recipientProvider(districtName)
 
      if (recipients.length === 0) {
        return res.status(200).json({
          dispatched: true,
          districtName,
          riskScore,
          riskBand,
          mode: smsGateway.mode,
          messagesSent: 0,
          warning: `No registered recipients found for "${districtName}".`,
          templates,
        })
      }
 
      const sendResults = await Promise.allSettled(
        recipients.map(async (recipient) => {
          const language = SUPPORTED_LANGUAGES.includes(recipient.preferredLanguage)
            ? recipient.preferredLanguage
            : DEFAULT_LANGUAGE
          const body = templates[language]
          const result = await smsGateway.send({ to: recipient.phoneNumber, body })
          return { phoneNumber: recipient.phoneNumber, language, ...result }
        }),
      )
 
      const results = sendResults.map((outcome, i) => {
        if (outcome.status === 'fulfilled') {
          return { ...outcome.value, success: true }
        }
        return {
          phoneNumber: recipients[i].phoneNumber,
          language: recipients[i].preferredLanguage,
          success: false,
          error: outcome.reason?.message || String(outcome.reason),
        }
      })
 
      const sentCount = results.filter((r) => r.success).length
      const failedCount = results.length - sentCount
      const perLanguageCounts = results.reduce((acc, r) => {
        acc[r.language] = (acc[r.language] || 0) + 1
        return acc
      }, {})
 
      return res.status(200).json({
        dispatched: true,
        districtName,
        riskScore,
        riskBand,
        mode: smsGateway.mode,
        messagesSent: sentCount,
        messagesFailed: failedCount,
        perLanguageCounts,
        templates,
        results,
      })
    } catch (err) {
      next(err)
    }
  })
 
  return router
}
 
module.exports = {
  createDispatchAlertsRouter,
  // Exported for direct unit testing / reuse.
  buildAlertTemplates,
  riskBandForScore,
  validateRequestBody,
  defaultRecipientProvider,
  HIGH_RISK_THRESHOLD,
  MEDIUM_RISK_THRESHOLD,
  SUPPORTED_LANGUAGES,
}
 