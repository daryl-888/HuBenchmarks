#!/usr/bin/env python3
"""
FlashVID × MotionBench — ovqwen2 (Qwen2 backbone).

FlashVID (ICLR 2026 Oral) uses pre-LLM token merging (ADTS + TSTM).
Model must be loaded first, then wrapped with flashvid().

Backbone: LLaVA-OV-7B-Qwen2 (llava-ov-7b-qwen2 weights)
Conv template: qwen_2
Params: retention=0.25, alpha=0.7, temporal_threshold=0.8, 32f

Requires: flashvid conda env
PYTHONPATH: /project/rhu/dpalfaro/code/FlashVID
"""
import argparse, json, os, re, sys
import torch
from tqdm import tqdm
# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT="\nAnswer with the option's letter from the given choices directly."

def load_model(model_path, retention_ratio=0.15, alpha=0.7, temporal_threshold=0.8):
    import sys as _sys
    _sys.path.insert(0,"/project/rhu/dpalfaro/code/FlashVID")
    from llava.model.builder import load_pretrained_model
    from flashvid import flashvid as apply_flashvid
    # attn_implementation="sdpa" is REQUIRED: the builder defaults to
    # flash_attention_2, which is not installed in this env (ImportError:
    # "flash_attn seems to be not installed"). sdpa is the verified-working path.
    t,m,ip,_=load_pretrained_model(model_path,None,"llava_qwen",
                                   attn_implementation="sdpa")
    m=apply_flashvid(m,retention_ratio=retention_ratio,alpha=alpha,
                     temporal_threshold=temporal_threshold,do_segment=True)
    m=m.cuda(); m.eval()
    return t,m,ip

def load_frames(vp,nf):
    import multiprocessing as _mp, queue as _queue
    def _w(p,n,q):
        try:
            import numpy as np; from decord import VideoReader,cpu
            vr=VideoReader(p,ctx=cpu(0)); idx=np.linspace(0,len(vr)-1,n,dtype=int)
            q.put(("ok",vr.get_batch(idx).asnumpy()))
        except Exception as e: q.put(("error",str(e)))
    q=_mp.Queue(); proc=_mp.Process(target=_w,args=(vp,nf,q)); proc.start()
    try: status,data=q.get(timeout=60)
    except _queue.Empty: proc.kill(); proc.join(5); raise RuntimeError(f"timeout:{vp}")
    proc.join(5)
    if proc.is_alive(): proc.kill(); proc.join(5)
    if status=="error": raise RuntimeError(f"decode:{vp} - {data}")
    from PIL import Image; return [Image.fromarray(f) for f in data]

@torch.inference_mode()
def run_inference(tok,model,ip,frames,question,conv_template="qwen_2"):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    msg=DEFAULT_IMAGE_TOKEN+"\n"+question+POST_PROMPT
    conv=conv_templates[conv_template].copy(); conv.append_message(conv.roles[0],msg); conv.append_message(conv.roles[1],None)
    ids=tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).cuda()
    imgs=ip.preprocess(frames,return_tensors="pt")["pixel_values"].to(dtype=model.dtype,device="cuda")
    w,h=frames[0].size
    out=model.generate(ids,images=[imgs],image_sizes=[(h,w)]*len(frames),modalities=["video"],do_sample=False,temperature=0,max_new_tokens=16,use_cache=True)
    return tok.batch_decode(out,skip_special_tokens=True)[0].strip()

def find_video(vp):
    for sd in ("self-collected","public-dataset"):
        f=os.path.join(VIDEO_BASE,sd,vp)
        if os.path.exists(f): return f
    return None
def score_prediction(pred,gt):
    gt=gt.strip().upper()
    if gt=="NA": return None
    m=re.search(r"\b([A-D])\b",pred.upper())
    return int((m.group(1) if m else pred.strip().upper()[:1])==gt)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model_path",required=True); p.add_argument("--meta_path",required=True)
    p.add_argument("--output_dir",required=True); p.add_argument("--num_frames",type=int,default=32)
    p.add_argument("--limit",type=int,default=None); p.add_argument("--conv_template",default="qwen_2")
    # accept both spellings: the sbatch historically passed --retention-ratio
    p.add_argument("--retention_ratio","--retention-ratio",type=float,default=0.15,
                   dest="retention_ratio",
                   help="fraction of visual tokens to KEEP (standardized: 0.15)")
    p.add_argument("--alpha",type=float,default=0.7)
    p.add_argument("--temporal_threshold",type=float,default=0.8)
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    print("Loading...",flush=True)
    tok,model,ip=load_model(a.model_path,retention_ratio=a.retention_ratio,
                            alpha=a.alpha,temporal_threshold=a.temporal_threshold)
    samples=[json.loads(l) for l in open(a.meta_path) if l.strip()]
    if a.limit: samples=samples[:a.limit]
    print(f"Evaluating {len(samples)} samples",flush=True)
    results,scores,pc=[],[],{}
    for i,s in enumerate(tqdm(samples,desc="Evaluating")):
        vp=find_video(s["video_path"]); q=s["qa"][0]["question"]; gt=s["qa"][0]["answer"]; qt=s.get("question_type","Unknown")
        pred=""
        if vp:
            try:
                fr=load_frames(vp,a.num_frames)
                pred=run_inference(tok,model,ip,fr,q,conv_template=a.conv_template)
            except Exception as e: print(f"  [WARN] {i}: {e}",file=sys.stderr); torch.cuda.empty_cache()
        sc=score_prediction(pred,gt)
        results.append({"idx":i,"video_path":s["video_path"],"question_type":qt,"ground_truth":gt,"prediction":pred,"correct":sc})
        if sc is not None:
            scores.append(sc); pc[qt]=pc.get(qt,{"correct":0,"total":0}); pc[qt]["total"]+=1; pc[qt]["correct"]+=sc
    with open(os.path.join(a.output_dir,"results.jsonl"),"w") as f:
        for r in results: f.write(json.dumps(r)+"\n")
    total=len(scores); correct=sum(scores); na=len(results)-total; acc=correct/total if total>0 else 0.0
    summary={"accuracy":acc,"correct":correct,"total_scoreable":total,"total_na_skipped":na,"total_samples":len(results),"model":a.model_path,"num_frames":a.num_frames,"conv_template":a.conv_template,"flashvid_params":{"enabled":True,"retention_ratio":a.retention_ratio,"alpha":a.alpha,"temporal_threshold":a.temporal_threshold},"per_category":pc}
    print(f"\nAccuracy: {correct}/{total}={acc:.4f} ({na} NA)",flush=True)
    for c in sorted(pc): print(f"  {c}: {pc[c]['correct']}/{pc[c]['total']}={pc[c]['correct']/pc[c]['total']:.4f}",flush=True)
    with open(os.path.join(a.output_dir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
    print(f"Results: {a.output_dir}/results.jsonl",flush=True); print(f"Summary: {a.output_dir}/summary.json",flush=True)
if __name__=="__main__": main()
