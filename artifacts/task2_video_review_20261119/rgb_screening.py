from pathlib import Path
import subprocess, json, hashlib
import cv2
import numpy as np
from main import load_config
from cube.detect import _white_pixels
from cube.stack_look import cap_color_masks
out=Path('artifacts/task2_video_review_20261119')
samples={'01-02-41':[0,10,15,20,30,40,45,50,60,70,75,76,78],
         '00-55-08':[0,10,20,30,40,50,60,70],
         '00-44-15':[0,30,60,120,180,240,250,260,270,280,290]}
keyframes={('01-02-41',15),('01-02-41',75),('01-02-41',76),('01-02-41',78),
           ('00-55-08',30),('00-55-08',70),('00-44-15',270)}
results=[]
colors=load_config('config/default.yaml')['colors']
for tag,times in samples.items():
 for t in times:
  cache=Path('/tmp/task2-review')/f'{tag}-{t}.png'
  if not cache.exists():
   subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',f'/home/dong/2026-11-18 {tag}.mkv',
                   '-frames:v','1','-vf','scale=1600:-1',str(cache)],check=True)
  screen=cv2.imread(str(cache))
  # This observed window occupies a 1280x720 camera viewport at this screenshot scale.
  camera=screen[88:808,24:1304].copy()
  valid=np.ones(camera.shape[:2],bool)
  valid[:300,:800]=False  # mask overlaid diagnostic text
  valid[590:]=False       # mask overlaid movement/result footer
  white=_white_pixels(camera,colors)>0
  red,blue=cap_color_masks(camera)
  red=(red>0)&valid;blue=(blue>0)&valid;white&=valid
  row=dict(video=f'/home/dong/2026-11-18 {tag}.mkv',timestamp_s=t,
           white_pixels=int(white.sum()),red_pixels=int(red.sum()),blue_pixels=int(blue.sum()),
           evaluated_pixels=int(valid.sum()),full_layer_detection='not_run_missing_depth_and_gravity')
  results.append(row)
  if (tag,t) in keyframes:
   cv2.imwrite(str(out/f'{tag}_{t:03d}s_screen.png'),screen)
   mask=np.zeros_like(camera)
   mask[white]=(210,210,210);mask[red]=(0,0,255);mask[blue]=(255,0,0)
   camera[~valid]=(25,25,25)
   pair=np.hstack([camera,mask])
   pair=cv2.copyMakeBorder(pair,40,0,0,0,cv2.BORDER_CONSTANT)
   cv2.putText(pair,f'{tag} +{t}s | CURRENT RGB ONLY | left: masked screenshot; right: white/red/blue masks',
               (15,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
   cv2.imwrite(str(out/f'{tag}_{t:03d}s_rgb.png'),pair)
 print(tag,len(times),'RGB samples checked',flush=True)
manifest=dict(scope='RGB-screening only; not a full-depth replay or accuracy evaluation',
              camera_crop_on_1600x1000_screen=[24,88,1304,808],
              exclusions=['camera top-left x<800,y<300 (HUD)','camera y>=590 (footer)'],
              limitations=['video is a screen recording with existing annotations',
                           'no original depth/gravity/intrinsics stream found',
                           'candidate color masks cannot determine layers, pose or reject background alone'],
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path('cube/detect.py'),Path('cube/stack_look.py'),Path('config/default.yaml')]},
              samples=results)
(out/'rgb_checks.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
