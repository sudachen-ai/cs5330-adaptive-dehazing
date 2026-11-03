"""Reproduce the validation-only blend, feature-ablation and small tuning study."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from dehaze import process, local_weights, save_rgb
from evaluate_dehazing import load_rgb, calculate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('splits/real_haze/validation.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('datasets/datasets_quarter_size'))
    parser.add_argument('--output', type=Path, default=Path('reports/suda_validation_study'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    frame = pd.read_csv(args.manifest, dtype=str)
    # Declare search and selection before seeing results. All variants share DCP.
    specs = [('adaptive_default', 15, -0.5, {})]
    specs += [('no_' + term, 15, -0.5, {term + '_coef': 0.0})
              for term in ('brightness', 'saturation', 'contrast', 'gradient')]
    specs += [('tune_p%d_b%s' % (patch, bias), patch, bias, {})
              for patch in (7, 15, 31) for bias in (-1.0, -0.5, 0.0)
              if (patch, bias) != (15, -0.5)]
    args.output.mkdir(parents=True)
    config = dict(manifest=str(args.manifest), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  objective='maximize mean per-image validation PSNR over both datasets; first listed wins ties',
                  alphas=[0, .25, .5, .75, 1], specs=specs,
                  dcp=dict(patch=15, omega=.95, t_min=.1, radius=20, eps=.001),
                  note='Patch search changes feature windows only; DCP parameters stay fixed. No test evaluation.',
                  source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                                 for n in ('dehaze.py','run_validation_study.py')})
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    frame.to_csv(args.output/'manifest.csv', index=False)
    records = []
    started = time.perf_counter()
    for i,row in frame.iterrows():
        image, gt = load_rgb(args.data_root/row.hazy_path), load_rgb(args.data_root/row.gt_path)
        result = process(image)
        outputs = {'original':image, 'dcp':result['dcp'], 'matched_fixed':result['matched_fixed']}
        for alpha in config['alphas']:
            outputs[f'fixed_{alpha:g}'] = (1-alpha)*image+alpha*result['dcp']
        for name,patch,bias,kwargs in specs:
            weights = result['weights'] if name == 'adaptive_default' else local_weights(image, patch=patch, bias=bias, **kwargs)[0]
            outputs[name] = (1-weights[...,None])*image+weights[...,None]*result['dcp']
        raw = result['raw_weights']
        outputs['no_weight_refinement'] = (1-raw[...,None])*image+raw[...,None]*result['dcp']
        for name,predicted in outputs.items():
            psnr,ssim = calculate_metrics(gt,predicted)
            records.append(dict(dataset=row.dataset,scene_id=row.scene_id,image=row.hazy_path,
                                method=name,psnr=psnr,ssim=ssim))
        print(f'{i+1}/{len(frame)} {row.dataset}/{row.scene_id}', flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(args.output/'image_metrics.csv',index=False)
    summary = metrics.groupby(['dataset','method']).agg(psnr=('psnr','mean'),ssim=('ssim','mean'),n=('psnr','size'))
    summary.to_csv(args.output/'dataset_summary.csv')
    overall = metrics.groupby('method')[['psnr','ssim']].mean()
    overall.to_csv(args.output/'overall_summary.csv')
    fixed = max([f'fixed_{a:g}' for a in config['alphas']],key=lambda n:overall.loc[n,'psnr'])
    adaptive = max([s[0] for s in specs if s[0]=='adaptive_default' or s[0].startswith('tune_')], key=lambda n:overall.loc[n,'psnr'])
    config.update(selected_fixed=fixed,selected_adaptive=adaptive,total_seconds=time.perf_counter()-started)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    pivot = metrics.pivot(index=['dataset','scene_id','image'],columns='method',values='psnr')
    delta = pivot['adaptive_default']-pivot[fixed]
    candidates = list(dict.fromkeys([delta.idxmax(),delta.idxmin()]))
    # Extremes illustrate observed PSNR differences; they are not representative samples.
    for case,key in enumerate(candidates):
        row = frame[frame.hazy_path==key[2]].iloc[0]
        image,gt = load_rgb(args.data_root/row.hazy_path),load_rgb(args.data_root/row.gt_path)
        result=process(image)
        a=float(fixed.split('_')[1]); tiles=[]
        for name,pic in [('hazy',image),('ground truth',gt),('DCP',result['dcp']),
                         (fixed,(1-a)*image+a*result['dcp']),('adaptive default',result['adaptive']),
                         ('weights',np.repeat(result['weights'][...,None],3,axis=2))]:
            h,w=pic.shape[:2]; scale=min(1,480/w); thumb=cv2.resize(pic,(round(w*scale),round(h*scale)))
            tile=np.zeros((thumb.shape[0]+30,thumb.shape[1],3),np.float32);tile[30:]=thumb
            cv2.putText(tile,name,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(1,1,1),1,cv2.LINE_AA);tiles.append(tile)
        board=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
        save_rgb(args.output/f'case_{case+1}.png',board)
    (args.output/'cases.json').write_text(json.dumps([dict(dataset=k[0],scene_id=k[1],image=k[2],adaptive_minus_tuned_fixed_psnr=float(delta.loc[k])) for k in candidates],indent=2)+'\n')
    print(overall.to_string());print('Selected:',fixed,adaptive)


if __name__=='__main__':
    main()
