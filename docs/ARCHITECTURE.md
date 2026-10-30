# Local research application architecture

The app is a single-machine, loopback-only Python service. A small responsive browser UI calls a JSON API served by the same process. A bounded in-process worker pool runs article extraction and inference jobs. SQLite stores article, image-reference, analysis, prediction, and job records on disk. Uploaded images are stored by generated opaque IDs below the app's own persistent storage directory.

Inference is delegated to the repository's existing NumPy model implementations. Text and image classifiers are independent tasks. There is no inference-time fusion model or image-text mismatch model available, so neither is synthesized from component scores. Reported evaluation metrics retain their dataset and split provenance and are not claims about arbitrary web content.

Article extraction only accepts public HTTP/HTTPS destinations, revalidates redirects, rejects private or reserved IPs, bounds time and response bytes, and checks robots rules. Extracted content remains plain text; the UI never renders source HTML. Users must edit/confirm the text-image pair before analysis.

The service binds to `127.0.0.1` by default. It is not intended for unauthenticated public deployment. The app does not run training or dataset preprocessing. Source monitoring is out of scope for the first local release and remains disabled/deferred.
