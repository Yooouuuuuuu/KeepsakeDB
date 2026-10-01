# Smoke: does Milvus Lite 3.x do dense + BM25 + hybrid on Windows, in-process, with zh text?
import os, shutil, time
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
import numpy as np

path = os.path.join(os.path.dirname(__file__), "out", f"smoke_{int(time.time())}.db")
for p in (path, path + ".lock"):
    if os.path.exists(p): os.remove(p)
if os.path.isdir(path): shutil.rmtree(path)
t0 = time.time()
c = MilvusClient(path)
print("client up", round(time.time()-t0, 2), "s")
s = c.create_schema()
s.add_field("id", DataType.INT64, is_primary=True)
s.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
s.add_field("dense", DataType.FLOAT_VECTOR, dim=8)
s.add_function(Function(name="bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))
ip = c.prepare_index_params()
ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
ip.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
c.create_collection("mem", schema=s, index_params=ip)
rng = np.random.default_rng(0)
docs = ["我養了一隻貓叫Mochi，牠最愛吃罐頭", "我最喜歡吃麻辣鍋，越辣越好", "我平常都騎 Gogoro 上班", "週末去十八尖山爬山"]
c.insert("mem", [{"id": i, "text": d, "dense": rng.random(8).tolist()} for i, d in enumerate(docs)])
print("bm25 only:", [(h["id"], round(h["distance"], 3)) for h in c.search("mem", data=["罐頭"], anns_field="sparse", limit=3, output_fields=["text"])[0]])
print("bm25 only (Gogoro):", [(h["id"], round(h["distance"], 3)) for h in c.search("mem", data=["gogoro"], anns_field="sparse", limit=3)[0]])
r1 = AnnSearchRequest(data=["罐頭"], anns_field="sparse", param={}, limit=3)
r2 = AnnSearchRequest(data=[rng.random(8).tolist()], anns_field="dense", param={}, limit=3)
h = c.hybrid_search("mem", [r1, r2], ranker=RRFRanker(60), limit=3, output_fields=["text"])
print("hybrid RRF:", [(x["id"], round(x["distance"], 4)) for x in h[0]])
c.close()
