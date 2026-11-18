"""Instrumentation only; optimization remains in the upstream Wan22Trainer.train loop."""
import json,math,os,time,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
class BenchmarkWindow:
    def __init__(self,t):
        self.t=t; self.rank=t.accelerator.process_index;self.device=t.accelerator.device
        self.root=Path(t.output_dir);self.smoke=os.environ.get('FASTWAM_PHASE')=='smoke'
        self.stop=min(10 if self.smoke else 249, t.max_steps);self.warmup=min(0 if self.smoke else 10, max(0, self.stop-1));self.world=t.accelerator.num_processes;self.effective_batch=t.batch_size*t.gradient_accumulation_steps*self.world
        self.rows=[];self.elapsed=None;self.samples=0;self.start=None;self.prof=None
        self.params=[p for p in t.model.parameters() if p.requires_grad]
        self.before=None
        signature=hashlib.sha256()
        for p in self.params:
            signature.update(p.detach().flatten()[:64].cpu().contiguous().view(torch.uint8).numpy().tobytes())
        self.initial_signature=signature.hexdigest()
        import logging
        assert t.log_every==1 and not logging.getLogger('fastwam.trainer').disabled
        engine=t.model
        assert not getattr(engine, 'is_pipe_parallel', False)
        assert dist.get_world_size()==self.world
        assert engine.zero_optimization_stage()==1 and engine.bfloat16_enabled()
        assert engine.train_micro_batch_size_per_gpu()==8 and engine.gradient_accumulation_steps()==t.gradient_accumulation_steps
        assert engine.dp_world_size==self.world and engine.mp_world_size==1 and engine.train_batch_size()==self.effective_batch
        assert t.batch_size==8 and t.gradient_accumulation_steps>=1 and t.num_workers==8
        assert t.resume is None and t.save_every==0 and t.eval_every==0
        # Actual official default horizon, never override max_steps to 250.
        actual=dict(parameter_dtypes=sorted({str(p.dtype) for p in self.params}),master_partition_dtypes=sorted({str(p.dtype) for p in getattr(engine.optimizer,'single_partition_of_fp32_groups',[])}),log_every=t.log_every,trainer_logger_enabled=True,engine_class=type(engine).__name__,distributed_backend=dist.get_backend(),actual_bf16=engine.bfloat16_enabled(),actual_zero=engine.zero_optimization_stage(),actual_micro_batch=engine.train_micro_batch_size_per_gpu(),actual_accumulation=engine.gradient_accumulation_steps(),actual_global_batch=engine.train_batch_size(),initial_parameter_sample_sha256=self.initial_signature,rank=self.rank,trainer_batch=t.batch_size,world=t.accelerator.num_processes,
                    optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in t.optimizer.param_groups],
                    scheduler_horizon=t.max_steps,lr_warmup=int(t.max_steps*.05),
                    scheduler=t.scheduler.state_dict(),trainable_numel=sum(p.numel() for p in self.params),
                    deepspeed=engine._config._param_dict,DP=engine.dp_world_size,TP=engine.mp_world_size,
                    PP=1,compiled_denoise=t.model.module.compile_training_denoise,vae_mode='eager')
        (self.root/f'actual.rank{self.rank}.json').write_text(json.dumps(actual,indent=2,default=str))
    def before_batch(self,step):
        if self.smoke:
            self.before=torch.cat([p.detach().flatten()[:16].clone() for p in self.params])
            if step==0 and self.rank==0:
                import torch.nn.functional as F
                self.sdpa_original=F.scaled_dot_product_attention
                self.sdpa_choices={}
                def record_choice(*args, **kwargs):
                    choice=int(torch._fused_sdp_choice(*args, **kwargs))
                    self.sdpa_choices[str(choice)]=self.sdpa_choices.get(str(choice),0)+1
                    return self.sdpa_original(*args, **kwargs)
                F.scaled_dot_product_attention=record_choice
        if step==self.warmup:
            dist.barrier();torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start=time.perf_counter()
    def after_update(self,step,sample,loss,norm):
        if step>self.warmup:self.samples+=int(sample['video'].shape[0])*self.t.gradient_accumulation_steps
        if step==self.stop:
            torch.cuda.synchronize(self.device);self.elapsed=time.perf_counter()-self.start
        if step==1:
            shapes={k:list(v.shape) for k,v in sample.items() if torch.is_tensor(v)}
            (self.root/f'input-shapes.rank{self.rank}.json').write_text(json.dumps(shapes,indent=2))
        norm=float(norm);value=float(loss.detach())
        ok=math.isfinite(value) and math.isfinite(norm) and not self.t.accelerator.optimizer_step_was_skipped
        changed=None
        if self.smoke:
            after=torch.cat([p.detach().flatten()[:16] for p in self.params]);changed=bool((after!=self.before).any())
            ok=ok and changed and bool(torch.stack([p.isfinite().all() for p in self.params]).all())
        assert ok, f'nonfinite/skipped/unchanged update rank={self.rank} step={step}'
        if self.smoke and step==1 and self.rank==0:
            import torch.nn.functional as F
            F.scaled_dot_product_attention=self.sdpa_original
            record={'method':'torch._fused_sdp_choice on actual first-step SDPA arguments; normal SDPA dispatch not overridden', 'choices':self.sdpa_choices,'backend_id_map':{'0':'MATH','1':'FLASH_ATTENTION','2':'EFFICIENT_ATTENTION','3':'CUDNN_ATTENTION','4':'OVERRIDEABLE'},'profiler_used':False}
            (self.root/'sdpa-selected-backends.json').write_text(json.dumps(record,indent=2))
        self.rows.append(dict(step=step,loss=value,grad_norm=norm,samples=int(sample['video'].shape[0]),changed=changed,finite=ok,lr=self.t.optimizer.param_groups[0]['lr']))
    def finish_if_done(self,step):
        if step!=self.stop:return False
        # All writes and reductions here occur after the synchronized end timestamp.
        result={'rank':self.rank,'seconds':self.elapsed,'measured_steps':self.stop-self.warmup,'samples':self.samples,
                'allocated':torch.cuda.max_memory_allocated(self.device),'reserved':torch.cuda.max_memory_reserved(self.device)}
        (self.root/f'window.rank{self.rank}.json').write_text(json.dumps(result,indent=2))
        (self.root/f'steps.rank{self.rank}.json').write_text(json.dumps(self.rows,indent=2))
        allr=[None]*self.world;dist.all_gather_object(allr,result)
        if self.rank==0:
            count=sum(x['samples'] for x in allr);duration=max(x['seconds'] for x in allr)
            assert count==self.effective_batch*(self.stop-self.warmup)
            summary={'valid':True,'phase':'smoke' if self.smoke else 'formal','warmup':self.warmup,'steps':self.stop-self.warmup,'samples':count,'window_s':duration,'samples_per_s':count/duration,'s_per_optimizer_step':duration/(self.stop-self.warmup),'ranks':allr,'scheduler_horizon':self.t.max_steps,'timing':'continuous monotonic window, includes batch wait/H2D/forward/backward/communication/update and normal official per-step logging; no added step barriers'}
            (self.root/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        return True
