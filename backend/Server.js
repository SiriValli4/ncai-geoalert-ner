'use strict';
 
const express = require('express');
const { createDispatchAlertsRouter } = require('./dispatchAlertsRoute');
 
const app = express();
app.use(express.json());
app.use('/api', createDispatchAlertsRouter());
 
// Basic error handler so an unexpected throw returns JSON, not an HTML stack trace.
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});
 
const PORT = process.env.PORT || 3000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`dispatch-alerts example server listening on :${PORT}`));
}
 
module.exports = app;
