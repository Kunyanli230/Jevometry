# Security and data handling

## Credentials

* Live access uses `TYPESAFE_API_KEY` from the process environment.  The value
  is never logged, stored, hashed or included in artifacts.
* `jevometry doctor` reports only whether the variable is set.
* `RecordingTransport` redacts `Authorization`, `Proxy-Authorization`,
  `X-API-Key` and `API-Key` headers before recording.
* `.env` files are not loaded automatically and are git-ignored.

## Data

* Only live adapters transmit rendered states and questions.
* User-declared redaction is applied before hashing, so the redacted payload is
  the analysis input and is reflected in fingerprints.
* Raw captures remain local.  Reports embed summaries, hashes and parameters,
  not sample text.
* Reports are self-contained: Plotly.js is inlined, no CDN is contacted and no
  telemetry is sent.  External text is HTML-escaped.

## Reporting a vulnerability

This repository has no public contact address yet; the owner must designate
one before public distribution.  Until then, report issues privately to the
maintainer of your checkout and do not open a public issue containing secrets
or customer data.

## Scope

Jevometry analyses declared models and recorded responses.  It does not modify
an analysed system's behaviour, does not call tools on its behalf and does not
provide a network service.
