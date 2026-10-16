# Live leaderboard screenshots

Source: [MTEB(Multilingual, v2), performance per task](https://mteb-leaderboard.hf.space/benchmark/MTEB%28Multilingual%2C%20v2%29?q=multilingual-e5-large&tab=perf_task).

Use the **second row**, `intfloat/multilingual-e5-large`; the first row is the distinct instruct model. The displayed scores are IN22GenBitextMining **76.75**, BelebeleRetrieval **77.91**, and STS22.v2 **64.30**.

The task-named PNGs are compact crops suitable for a PR. The `-page.png` variants retain surrounding webpage context. Only rectangular cropping was applied; no scores, labels, or webpage content were changed. Browser tabs and bookmarks are excluded. Capture metadata is in `capture.json`.

Suggested PR caption:

> Live MTEB(Multilingual, v2) scores for intfloat/multilingual-e5-large: IN22GenBitextMining 76.75, BelebeleRetrieval 77.91, and STS22.v2 64.30. These displayed values agree with the direct/tagged aggregation in the local reproduction. The inflated scores discussed in this PR come from reproducing the old tagged-cache splitting path locally; these screenshots do not show inflated scores on the live leaderboard.
