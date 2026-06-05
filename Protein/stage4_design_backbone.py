# -*- coding: utf-8 -*-







"""Stage4 multi-candidate backbone design and ranking."""







import os, sys, csv, math, shutil, argparse, subprocess







import numpy as np







import torch







import torch.nn.functional as F















sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))







from geometry_utils import HELIX_PHI, HELIX_PSI, BETA_PHI, BETA_PSI, sincos_to_torsion







from backbone_losses import combined_backbone_loss







from stage4_model import TorsionGenerator, TOPOLOGY_FAMILIES
from stage3_model import DockingModel







from stage4_generate import (normalize_torsion_sincos, generate_backbone, write_pdb,







    validate_backbone, ss_torsion_loss, ss_target_to_torsion_sincos, load_compatible_state_dict)















COIL_PHI, COIL_PSI = -75.0, 145.0























def set_seed(seed):







    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)























def load_receptor_npz(path, K, seq_len, device):







    with np.load(path, allow_pickle=True) as d:







        xs=d['xs'].astype(np.float32); ns=d['ns'].astype(np.float32)







        pc=d['patch_centers'].astype(np.float32); knn=d['patch_knn_idx'].astype(np.int64)







        order=d['patch_order'].astype(np.int64)







    if knn.shape[1] < K:







        knn=np.concatenate([knn, np.tile(knn[:,-1:], (1,K-knn.shape[1]))], 1)







    else: knn=knn[:,:K]







    sel=order[:min(len(pc), seq_len)]







    idx=knn[sel]; ctrs=pc[sel]







    feats=np.concatenate([xs[idx]-ctrs[:,None,:], ns[idx]], -1).astype(np.float32)







    return (torch.from_numpy(feats).unsqueeze(0).to(device),







            torch.from_numpy(ctrs).unsqueeze(0).to(device),







            torch.zeros(1, len(sel), dtype=torch.bool, device=device), ctrs)























def build_generator(args, device):







    ckpt=torch.load(args.stage4_ckpt, map_location=device)







    max_res=max(args.max_residues, int(ckpt.get('args',{}).get('max_residues', args.max_residues)))







    m=TorsionGenerator(args.d_model,args.nhead,args.nlayers_surf,args.nlayers_gen,args.K,max_res).to(device)







    if args.stage3_ckpt: m.load_from_stage3(args.stage3_ckpt, device=device)







    load_compatible_state_dict(m, ckpt['model'], label=args.stage4_ckpt)







    m.eval(); return m






















def build_stage3_guidance(args, device):
    if not args.stage3_ckpt:
        return None
    ckpt_path=args.stage3_ckpt
    if not os.path.isfile(ckpt_path):
        print(f'[WARN] Stage3 ckpt not found for guidance: {ckpt_path}')
        return None
    model=DockingModel(d_model=args.d_model,nhead=args.nhead,nlayers_surf=args.nlayers_surf,nlayers_bb=4,K=args.K).to(device)
    ckpt=torch.load(ckpt_path,map_location=device)
    state=ckpt.get('model',ckpt)
    current=model.state_dict()
    matched={k:v for k,v in state.items() if k in current and tuple(current[k].shape)==tuple(v.shape)}
    model.load_state_dict(matched,strict=False)
    model.eval()
    print(f'[Stage3 guidance] loaded matched={len(matched)} from {ckpt_path}')
    return model


def predict_stage3_pocket(stage3_model, rec_feats, rec_centers, rec_pad, receptor_points_np=None):
    if receptor_points_np is None:
        receptor_points_np=rec_centers[0].detach().cpu().numpy().astype(np.float32)
    pts=np.asarray(receptor_points_np,dtype=np.float32)
    surf_center=pts.mean(0).astype(np.float32)
    surf_spread=float(np.sqrt(((pts-surf_center[None,:])**2).sum(1).mean()))
    if stage3_model is None:
        return {'center': surf_center, 'radius': surf_spread, 'source': 'surface_fallback', 'confidence': 0.0}
    with torch.no_grad():
        rec_tokens=stage3_model.rec_surf_encoder(rec_feats, rec_centers, key_padding_mask=rec_pad)
        center,radius=stage3_model.pocket_head(rec_tokens, rec_pad)
    raw_center=center[0].detach().cpu().numpy().astype(np.float32)
    raw_radius=float(radius[0,0].detach().cpu().item())
    dist=float(np.linalg.norm(raw_center-surf_center))
    reliable=(raw_radius>2.0 and raw_radius<1.5*surf_spread and dist<2.0*surf_spread)
    if reliable:
        confidence=1.0
        eff_center=raw_center
        eff_radius=raw_radius
        source='stage3'
    else:
        confidence=0.25
        eff_center=surf_center
        eff_radius=surf_spread
        source='surface_fallback_stage3_unreliable'
    print(f'[Stage3 guidance] raw_center={raw_center.tolist()} raw_radius={raw_radius:.2f} source={source} confidence={confidence:.2f}')
    return {'center': eff_center.astype(np.float32), 'radius': float(eff_radius), 'raw_center': raw_center, 'raw_radius': raw_radius, 'source': source, 'confidence': confidence}



def predict_stage4_pocket(model, rec_feats, rec_centers, rec_pad, num_residues, receptor_points_np=None):
    if receptor_points_np is None:
        receptor_points_np=rec_centers[0].detach().cpu().numpy().astype(np.float32)
    pts=np.asarray(receptor_points_np,dtype=np.float32)
    surf_center=pts.mean(0).astype(np.float32)
    surf_spread=float(np.sqrt(((pts-surf_center[None,:])**2).sum(1).mean()))
    with torch.no_grad():
        *_,pocket_pred,hotspot_logits,topology_logits=model(rec_feats,rec_centers,rec_pad,num_residues)
    probs=torch.sigmoid(hotspot_logits[0]).detach().cpu().numpy().astype(np.float32)
    valid=~rec_pad[0].detach().cpu().numpy().astype(bool)
    probs=np.where(valid,probs,-1.0)
    top_idx=np.argsort(-probs)[:max(1,min(32,valid.sum()))]
    top_centers=rec_centers[0].detach().cpu().numpy().astype(np.float32)[top_idx]
    raw_delta=pocket_pred[0,:3].detach().cpu().numpy().astype(np.float32)
    raw_center=(surf_center+raw_delta).astype(np.float32)
    raw_radius=float(F.softplus(pocket_pred[0,3:4]).detach().cpu().item())
    dist=float(np.linalg.norm(raw_center-surf_center))
    reliable=(raw_radius>2.0 and raw_radius<1.5*surf_spread and dist<2.0*surf_spread)
    source='stage4_relative' if reliable else 'surface_fallback_stage4_unreliable'
    center=raw_center if reliable else surf_center
    radius=raw_radius if reliable else surf_spread
    confidence=1.0 if reliable else 0.25
    hot_center=top_centers[:min(8,len(top_centers))].mean(0).astype(np.float32)
    print(f'[Stage4 pocket] raw_delta={raw_delta.tolist()} raw_center={raw_center.tolist()} raw_radius={raw_radius:.2f} hotspot_center={hot_center.tolist()} top_hotspot={float(probs[top_idx[0]]):.3f} source={source} confidence={confidence:.2f}')
    return {'center': center.astype(np.float32), 'radius': float(radius), 'raw_center': raw_center, 'raw_delta': raw_delta, 'raw_radius': raw_radius, 'source': source, 'confidence': confidence, 'hotspot_probs': probs, 'hotspot_top_idx': top_idx.astype(np.int64), 'hotspot_centers': top_centers, 'hotspot_center': hot_center}


def nearest_surface_normal_to_center(center, surf_np):
    surf_center=surf_np.mean(0)
    idx=int(np.argmin(np.linalg.norm(surf_np-center[None,:],axis=1)))
    anchor=surf_np[idx]
    normal=anchor-surf_center
    normal=normal/(np.linalg.norm(normal)+1e-8)
    return anchor.astype(np.float32), normal.astype(np.float32)


def place_backbone_predicted_pocket(bb, surf_np, pocket_info, cand_idx, contact_offset=6.0, jitter=2.0):
    bb=bb.copy()
    center=np.asarray(pocket_info['center'],dtype=np.float32)
    anchor,normal=nearest_surface_normal_to_center(center,surf_np)
    R=random_rotation(cand_idx+101, dtype=bb.dtype)
    ca=bb[:,1,:]
    flat=bb.reshape(-1,3)-ca.mean(0,keepdims=True)
    bb=(flat@R.T).reshape(bb.shape)
    ca=bb[:,1,:]
    interface_idx=len(ca)//2
    target=center
    rng=np.random.default_rng(9876+cand_idx*31)
    if jitter>0:
        target=target+rng.normal(0.0,jitter,size=3).astype(np.float32)
    # Keep the candidate outside the receptor surface if the predicted center falls too deep.
    if np.linalg.norm(target-anchor) < contact_offset:
        target=anchor+normal*contact_offset
    bb=bb+(target-ca[interface_idx])[None,None,:]
    return bb.astype(np.float32)

def choose_auto_topology(pocket_info, receptor_points_np, cand_idx, exploration_frac=0.35):
    pts=np.asarray(receptor_points_np,dtype=np.float32)
    center=pts.mean(0)
    spread=float(np.sqrt(((pts-center[None,:])**2).sum(1).mean()))
    radius=pocket_info['radius'] if pocket_info is not None else spread
    families=['helical_peptide','helix_loop_helix','three_helix_bundle','four_helix_bundle','beta_hairpin','coil_peptide']
    # Every auto batch must explore every topology at least once per cycle.
    if cand_idx < len(families):
        return families[cand_idx]
    # Deterministic exploration slots keep diversity even for large receptors.
    period=max(5, int(round(1.0/max(exploration_frac,1e-3))))
    if cand_idx % period == 0:
        return families[(cand_idx // period) % len(families)]
    # Receptor-conditioned exploitation weights, but never zero for any family.
    if radius < 7.0:
        weighted=['helical_peptide']*3 + ['helix_loop_helix']*5 + ['three_helix_bundle']*1 + ['four_helix_bundle'] + ['beta_hairpin'] + ['coil_peptide']
    elif radius < 12.0:
        weighted=['helical_peptide']*2 + ['helix_loop_helix']*5 + ['three_helix_bundle']*2 + ['four_helix_bundle'] + ['beta_hairpin'] + ['coil_peptide']
    else:
        weighted=['helical_peptide']*2 + ['helix_loop_helix']*4 + ['three_helix_bundle']*2 + ['four_helix_bundle']*2 + ['beta_hairpin'] + ['coil_peptide']
    return weighted[cand_idx % len(weighted)]



def topology_ss(topology, L, device):







    ss=torch.full((1,L),2,dtype=torch.long,device=device)







    if topology in ['helix','helical_peptide']:







        ss[:]=0







    elif topology=='helix_loop_helix':







        loop=max(3,min(4,L//10)); h=(L-loop)//2; ss[:,:h]=0; ss[:,h:h+loop]=2; ss[:,h+loop:]=0







    elif topology=='three_helix_bundle':







        loop=max(3,L//10); h=max(5,(L-2*loop)//3)







        ss[:,:h]=0; ss[:,h:h+loop]=2; ss[:,h+loop:h+loop+h]=0







        ss[:,h+loop+h:h+loop+h+loop]=2; ss[:,h+loop+h+loop:]=0







    elif topology=='four_helix_bundle':

        loop=max(3,L//12); h=max(4,(L-3*loop)//4)

        ss[:,:h]=0; ss[:,h:h+loop]=2; ss[:,h+loop:h+loop+h]=0

        ss[:,h+loop+h:h+loop+h+loop]=2; ss[:,h+loop+h+loop:h+loop+h+loop+h]=0

        ss[:,h+loop+h+loop+h:h+loop+h+loop+h+loop]=2; ss[:,h+loop+h+loop+h+loop:]=0

    elif topology=='beta_hairpin':







        loop=max(3,L//5); e=(L-loop)//2; ss[:,:e]=1; ss[:,e:e+loop]=2; ss[:,e+loop:]=1







    elif topology=='coil_peptide':

        ss[:]=2



    elif topology=='mixed':







        h=max(6,L//3); e=max(h+4,2*L//3); ss[:,:h]=0; ss[:,h:e]=2; ss[:,e:]=1







    else: raise ValueError(topology)







    return ss























def sample_torsion(base, ss, noise_deg, blend):







    target=ss_target_to_torsion_sincos(ss, fallback=base)







    x=normalize_torsion_sincos((1-blend)*target + blend*normalize_torsion_sincos(base))







    if noise_deg>0:







        deg=sincos_to_torsion(x); n=torch.randn_like(deg)*noise_deg; n[...,2]*=0.20







        deg=deg+n; rad=deg*math.pi/180.0; x=torch.cat([torch.sin(rad), torch.cos(rad)], -1)







    return normalize_torsion_sincos(x)























def ca_repulsion(bb, min_dist):







    CA=bb[:,:,1,:]; B,L,_=CA.shape; loss=bb.new_tensor(0.)







    for b in range(B):







        d=torch.cdist(CA[b:b+1], CA[b:b+1]).squeeze(0); ids=torch.arange(L,device=bb.device)







        mask=((ids[:,None]-ids[None,:]).abs()>=3) & torch.triu(torch.ones(L,L,dtype=torch.bool,device=bb.device),1)







        if mask.any(): loss=loss+F.relu(min_dist-d[mask]).pow(2).mean()







    return loss/B























def compact_loss(bb, target_rg, min_contacts, contact_dist):







    CA=bb[:,:,1,:]; center=CA.mean(1,keepdim=True)







    rg=torch.sqrt(((CA-center)**2).sum(-1).mean(1).clamp(min=1e-8))







    rg_loss=((rg-target_rg)/max(target_rg,1.)).pow(2).mean()







    d=torch.cdist(CA,CA)[0]; L=CA.shape[1]; ids=torch.arange(L,device=bb.device)







    mask=(ids[:,None]-ids[None,:]).abs()>=6







    contacts=((d<contact_dist)&mask).float().sum()/2







    return rg_loss + F.relu(float(min_contacts)-contacts).pow(2)























def helix_segments_from_ss(ss_row, min_len=4):
    vals=ss_row.detach().cpu().tolist()
    segs=[]; start=None
    for i,v in enumerate(vals):
        if v==0 and start is None:
            start=i
        elif v!=0 and start is not None:
            if i-start>=min_len: segs.append((start,i))
            start=None
    if start is not None and len(vals)-start>=min_len: segs.append((start,len(vals)))
    return segs


def helix_bundle_packing_loss(bb, ss, target_dist=10.0, min_dist=6.0, max_dist=13.0, axis_abs_cos=0.65):
    CA=bb[0,:,1,:]
    segs=helix_segments_from_ss(ss[0])
    if len(segs)<2:
        return bb.new_tensor(0.)
    centers=[]; axes=[]
    for a,b in segs:
        pts=CA[a:b]
        centers.append(pts.mean(0))
        axis=pts[-1]-pts[0]
        axes.append(axis/(axis.norm()+1e-8))
    loss=bb.new_tensor(0.)
    n=0
    for i in range(len(segs)):
        for j in range(i+1,len(segs)):
            d=(centers[i]-centers[j]).norm()
            loss=loss+((d-target_dist)/target_dist).pow(2)
            loss=loss+F.relu(min_dist-d).pow(2)+F.relu(d-max_dist).pow(2)
            cos=(axes[i]*axes[j]).sum().abs()
            loss=loss+F.relu(axis_abs_cos-cos).pow(2)
            n+=1
    loss=loss/max(1,n)
    if len(segs)>=3:
        v1=centers[1]-centers[0]; v2=centers[2]-centers[0]
        area=torch.linalg.cross(v1,v2).norm()/2.0
        loss=loss+F.relu(18.0-area).pow(2)/100.0
    return loss


def surface_contact_loss(bb, surf, target_frac, min_d, max_d):







    CA=bb[:,:,1,:]; d=torch.cdist(CA, surf.unsqueeze(0)).min(-1).values







    frac=((d>=min_d)&(d<=max_d)).float().mean(1)







    return F.relu(target_frac-frac).mean()+F.relu(min_d-d).pow(2).mean()+0.05*F.relu(d-max_d).mean()























def native_ca_target(native_info, n_residues, align_target='ligand'):
    if native_info is None:
        return None
    if align_target == 'interface':
        pts = native_info['iface_ca']
    else:
        pts = native_info['lig_ca']
    return resample_polyline(pts, n_residues).astype(np.float32)


def native_shape_loss(bb, native_info, args):
    if native_info is None or getattr(args, 'w_native_shape', 0.0) <= 0:
        return bb.new_tensor(0.)
    target_np = native_ca_target(native_info, bb.shape[1], args.native_align_target)
    target = torch.from_numpy(target_np).to(device=bb.device, dtype=bb.dtype).unsqueeze(0)
    P = bb[:, :, 1, :]
    Pc = P - P.mean(dim=1, keepdim=True)
    Qc = target - target.mean(dim=1, keepdim=True)
    H = torch.matmul(Pc.transpose(1, 2), Qc)
    U, _, Vh = torch.linalg.svd(H)
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
    det = torch.linalg.det(R)
    if (det < 0).any():
        Vh = Vh.clone()
        Vh[det < 0, -1, :] *= -1
        R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
    aligned = torch.matmul(Pc, R)
    rmsd = torch.sqrt(((aligned - Qc) ** 2).sum(dim=-1).mean(dim=-1).clamp(min=1e-8))
    return rmsd.mean()


def refine(x0, ss, surf, args, native_info=None):







    p=torch.nn.Parameter(x0.detach().clone()); opt=torch.optim.Adam([p], lr=args.refine_lr)







    target_rg=args.target_rg if args.target_rg>0 else max(6.0, 2.2*(args.num_residues**(1/3)))







    for _ in range(args.refine_steps):







        opt.zero_grad(); x=normalize_torsion_sincos(p); bb=generate_backbone(x)







        losses=combined_backbone_loss(bb,None,args.w_bond,args.w_angle,args.w_omega,args.w_rama,args.w_clash)







        total=losses['total']+args.w_ss*ss_torsion_loss(x,ss)+args.w_rep*ca_repulsion(bb,args.repulsion_dist)







        total=total+args.w_compact*compact_loss(bb,target_rg,args.min_long_contacts,args.long_contact_dist)
        total=total+args.w_pack*helix_bundle_packing_loss(bb,ss,args.pack_target_dist,args.pack_min_dist,args.pack_max_dist,args.pack_axis_abs_cos)







        total=total+args.w_contact*surface_contact_loss(bb,surf,args.target_contact_frac,args.surface_min_dist,args.surface_max_dist)
        total=total+args.w_native_shape*native_shape_loss(bb,native_info,args)







        total.backward(); opt.step()







    with torch.no_grad():







        x=normalize_torsion_sincos(p); bb=generate_backbone(x)







    return bb.detach()
























def build_ideal_helix(length, radius=2.28, rise=1.50, phase=0.0, dtype=np.float32):
    theta=np.deg2rad(100.0)
    ca=[]
    for i in range(length):
        a=phase+i*theta
        ca.append([radius*np.cos(a), radius*np.sin(a), i*rise])
    return np.asarray(ca,dtype=dtype)


def frame_from_ca(ca):
    L=len(ca)
    N=np.zeros((L,3),dtype=np.float32); C=np.zeros((L,3),dtype=np.float32)
    for i in range(L):
        if i==0: t=ca[1]-ca[0]
        elif i==L-1: t=ca[-1]-ca[-2]
        else: t=ca[i+1]-ca[i-1]
        t=t/(np.linalg.norm(t)+1e-8)
        radial=np.array([ca[i,0], ca[i,1], 0.0],dtype=np.float32)
        if np.linalg.norm(radial)<1e-6: radial=np.array([1.,0.,0.],dtype=np.float32)
        radial=radial/(np.linalg.norm(radial)+1e-8)
        nvec=np.cross(t, radial); nvec=nvec/(np.linalg.norm(nvec)+1e-8)
        N[i]=ca[i]-1.45*t+0.18*nvec
        C[i]=ca[i]+1.52*t+0.12*nvec
    return np.stack([N,ca,C],axis=1).astype(np.float32)


def connect_segments_linear(bb_segments, loop_len=3):
    parts=[]
    for si,seg in enumerate(bb_segments):
        if si>0:
            prev=parts[-1][-1]
            nxt=seg[0]
            p0=prev[1]; p1=nxt[1]
            chord=p1-p0
            chord_len=np.linalg.norm(chord)+1e-8
            direction=chord/chord_len
            ref=np.array([0.,0.,1.],dtype=np.float32)
            if abs(np.dot(direction,ref))>0.85:
                ref=np.array([0.,1.,0.],dtype=np.float32)
            perp=np.cross(direction,ref); perp=perp/(np.linalg.norm(perp)+1e-8)
            best_h=0.0; best_err=1e9
            for h in np.linspace(0.0, max(12.0, 4.5*(loop_len+1)), 181):
                pts=[p0]
                for k in range(1,loop_len+1):
                    t=k/(loop_len+1)
                    pts.append((1-t)*p0+t*p1+perp*np.sin(np.pi*t)*h)
                pts.append(p1)
                ds=[np.linalg.norm(pts[k+1]-pts[k]) for k in range(len(pts)-1)]
                err=max(abs(float(x)-3.8) for x in ds)+0.10*np.std(ds)
                if err<best_err:
                    best_err=err; best_h=h
            loop=[]
            for k in range(1,loop_len+1):
                t=k/(loop_len+1)
                ca=(1-t)*p0+t*p1+perp*np.sin(np.pi*t)*best_h
                local=prev.copy()
                local_shift=ca-local[1]
                local=local+local_shift[None,:]
                loop.append(local)
            parts.append(np.stack(loop,axis=0).astype(np.float32))
        parts.append(seg)
    return np.concatenate(parts,axis=0)



def build_ideal_strand(length, phase=0.0, rise=3.25, step=1.45, dtype=np.float32):
    ca=[]
    for i in range(length):
        x=(( -1.0) ** i) * 0.6
        y=np.cos(phase + i * 2.2) * 0.25
        z=i*rise
        ca.append([x, y, z])
    return np.asarray(ca, dtype=dtype)


def family_segments(family, L):
    if family in ('helical_peptide', 'helix'):
        return [L], []
    if family == 'helix_loop_helix':
        loop = max(3, min(4, L // 10))
        h = max(5, (L - loop) // 2)
        return [h, L - h - loop], [loop]
    if family == 'three_helix_bundle':
        loop = max(3, L // 10)
        h = max(5, (L - 2 * loop) // 3)
        return [h, h, L - 2 * h - 2 * loop], [loop, loop]
    if family == 'four_helix_bundle':
        loop = max(3, L // 12)
        h = max(4, (L - 3 * loop) // 4)
        return [h, h, h, L - 3 * h - 3 * loop], [loop, loop, loop]
    if family == 'beta_hairpin':
        loop = max(3, L // 6)
        e = max(5, (L - loop) // 2)
        return [e, L - e - loop], [loop]
    raise ValueError(f'Unsupported family: {family}')


def build_parametric_bundle(L, family='three_helix_bundle', cand_idx=0, seed=2026):
    rng = np.random.default_rng(seed + cand_idx * 7919)
    seg_lens, loop_lens = family_segments(family, L)
    nseg = len(seg_lens)
    bundle_radius = float(rng.uniform(5.0, 6.8))
    phases = rng.uniform(0, 2 * np.pi, size=nseg)
    segs = []
    for j, n in enumerate(seg_lens):
        if family == 'beta_hairpin':
            ca = build_ideal_strand(n, phase=phases[j])
        else:
            ca = build_ideal_helix(n, phase=phases[j])
            if nseg >= 2 and j % 2 == 1:
                ca = ca[::-1].copy()
        angle = 2 * np.pi * j / max(1, nseg) + rng.uniform(-0.18, 0.18)
        center = np.array([
            bundle_radius * np.cos(angle),
            bundle_radius * np.sin(angle),
            rng.uniform(-1.5, 1.5),
        ], dtype=np.float32)
        ca = ca - ca.mean(0, keepdims=True)
        if family != 'beta_hairpin':
            tilt = float(rng.uniform(-0.25, 0.25))
            ca[:, 0] += tilt * ca[:, 2] * np.cos(angle + np.pi / 2)
            ca[:, 1] += tilt * ca[:, 2] * np.sin(angle + np.pi / 2)
        ca = ca + center[None, :]
        segs.append(frame_from_ca(ca))
    bb = connect_segments_linear(segs, loop_len=loop_lens[0] if loop_lens else 0)
    if len(bb) > L:
        bb = bb[:L]
    elif len(bb) < L:
        pad = np.repeat(bb[-1:, :, :], L - len(bb), axis=0)
        for k in range(len(pad)):
            pad[k] += np.array([0, 0, 3.8 * (k + 1)], dtype=np.float32)
        bb = np.concatenate([bb, pad], axis=0)
    bb = bb - bb[:, 1, :].mean(0, keepdims=True)[:, None, :]
    return bb.astype(np.float32)

def load_template_backbone(pdb_path, start_res=1, length=48, chain_id=None):
    residues={}
    for l in open(pdb_path):
        if not l.startswith('ATOM'):
            continue
        atom=l[12:16].strip(); chain=l[21].strip(); resi=int(l[22:26])
        if chain_id and chain != chain_id:
            continue
        if atom in ('N','CA','C'):
            residues.setdefault(resi,{})[atom]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])],dtype=np.float32)
    keys=[k for k in sorted(residues) if k>=start_res and all(a in residues[k] for a in ('N','CA','C'))]
    keys=keys[:length]
    if len(keys) < length:
        raise ValueError(f'template has only {len(keys)} complete residues, need {length}')
    bb=np.stack([[residues[k]['N'], residues[k]['CA'], residues[k]['C']] for k in keys],axis=0)
    bb=bb-bb[:,1,:].mean(axis=0,keepdims=True)[:,None,:]
    return bb.astype(np.float32)


def maybe_mirror_backbone(bb, mirror_axis):
    if mirror_axis == 'none':
        return bb
    bb=bb.copy()
    axis={'x':0,'y':1,'z':2}[mirror_axis]
    bb[...,axis] *= -1.0
    return bb


def random_rotation(seed, dtype=np.float32):



    rng=np.random.default_rng(seed)



    q=rng.normal(size=4); q=q/(np.linalg.norm(q)+1e-8)



    w,x,y,z=q



    R=np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],



                [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],



                [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=dtype)



    return R











def place_backbone_near_surface(bb, surf_np, cand_idx, contact_offset=6.0):



    bb=bb.copy()



    ca=bb[:,1,:]



    surf_center=surf_np.mean(0)



    anchor=surf_np[(cand_idx * 37 + 17) % len(surf_np)]



    normal=anchor-surf_center



    normal=normal/(np.linalg.norm(normal)+1e-8)



    R=random_rotation(cand_idx+17, dtype=bb.dtype)



    center=ca.mean(0,keepdims=True)



    flat=bb.reshape(-1,3)-center



    bb=(flat@R.T).reshape(bb.shape)



    ca=bb[:,1,:]



    interface_idx=len(ca)//2



    target=anchor+normal*contact_offset



    bb=bb+(target-ca[interface_idx])[None,None,:]



    return bb










def read_pdb_ca_chains(pdb_path, chains):
    chains=set(chains)
    ca=[]
    for l in open(pdb_path):
        if not l.startswith('ATOM') or l[12:16].strip()!='CA':
            continue
        ch=l[21].strip()
        if ch not in chains:
            continue
        ca.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    if not ca:
        raise ValueError(f'No CA atoms found for chains {chains} in {pdb_path}')
    return np.asarray(ca,dtype=np.float32)


def resample_polyline(points, n):
    points=np.asarray(points,dtype=np.float32)
    if len(points)==n:
        return points.copy()
    d=np.linalg.norm(points[1:]-points[:-1],axis=1)
    s=np.concatenate([[0.0],np.cumsum(d)])
    if s[-1] < 1e-6:
        return np.repeat(points[:1],n,axis=0)
    t=np.linspace(0.0,s[-1],n)
    out=[]
    for x in t:
        j=int(np.searchsorted(s,x,side='right')-1)
        j=max(0,min(j,len(points)-2))
        u=(x-s[j])/(s[j+1]-s[j]+1e-8)
        out.append((1-u)*points[j]+u*points[j+1])
    return np.asarray(out,dtype=np.float32)


def kabsch(P,Q):
    P=np.asarray(P,dtype=np.float32); Q=np.asarray(Q,dtype=np.float32)
    Pc=P-P.mean(0,keepdims=True); Qc=Q-Q.mean(0,keepdims=True)
    H=Pc.T@Qc
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1
        R=Vt.T@U.T
    return R.astype(np.float32), P.mean(0).astype(np.float32), Q.mean(0).astype(np.float32)


def build_native_guidance(args):
    if not args.native_pdb:
        return None
    rec=read_pdb_ca_chains(args.native_pdb,args.native_receptor_chains)
    lig=read_pdb_ca_chains(args.native_pdb,args.native_ligand_chains)
    d=np.linalg.norm(lig[:,None,:]-rec[None,:,:],axis=-1)
    mask=d.min(1)<=args.native_contact_cutoff
    iface=lig[mask]
    if len(iface)<max(6,args.num_residues//4):
        iface=lig
    if args.num_residues != len(lig):
        print(f'[Native guidance] note: num_residues={args.num_residues} differs from native ligand length={len(lig)}; target CA trace will be resampled')
    print(f'[Native guidance] ligand_ca={len(lig)} interface_ca={len(iface)} receptor_ca={len(rec)}')
    return {'rec_ca':rec,'lig_ca':lig,'iface_ca':iface}


def place_backbone_native_guided(bb,native_info,cand_idx,jitter=1.5,align_target='ligand',reverse='both'):
    bb=bb.copy()
    ca=bb[:,1,:]
    base_target=native_ca_target(native_info,len(ca),align_target)
    candidates=[]
    if reverse in ('auto','both'):
        targets=[base_target, base_target[::-1].copy()]
    elif reverse == 'yes':
        targets=[base_target[::-1].copy()]
    else:
        targets=[base_target]
    for target in targets:
        R,src_c,tgt_c=kabsch(ca,target)
        placed=((bb.reshape(-1,3)-src_c[None,:])@R.T).reshape(bb.shape)+tgt_c[None,None,:]
        placed_ca=placed[:,1,:]
        rmsd=float(np.sqrt(((placed_ca-target)**2).sum(1).mean()))
        candidates.append((rmsd,placed,target))
    candidates.sort(key=lambda x:x[0])
    bb=candidates[0][1]
    target=candidates[0][2]
    if jitter>0:
        rng=np.random.default_rng(12345+cand_idx*97)
        bb=bb+rng.normal(0.0,jitter,size=(1,1,3)).astype(np.float32)
    return bb.astype(np.float32), target.astype(np.float32)


def smooth_trace_displacement(disp, rounds=2):
    disp=np.asarray(disp,dtype=np.float32).copy()
    rounds=max(0,int(rounds))
    for _ in range(rounds):
        if len(disp)<=2:
            break
        new=disp.copy()
        new[1:-1]=0.25*disp[:-2]+0.50*disp[1:-1]+0.25*disp[2:]
        disp=new
    return disp.astype(np.float32)


def project_backbone_to_native_trace(bb, target_ca, strength=0.0, smooth_rounds=2):
    if strength<=0 or target_ca is None:
        return bb.astype(np.float32)
    strength=float(np.clip(strength,0.0,1.0))
    target=np.asarray(target_ca,dtype=np.float32)
    if len(target)!=len(bb):
        target=resample_polyline(target,len(bb))
    ca=bb[:,1,:]
    disp=(target-ca)*strength
    disp=smooth_trace_displacement(disp,smooth_rounds)
    return (bb+disp[:,None,:]).astype(np.float32)


def relax_projected_backbone(bb, target_ca=None, iterations=80, lr=0.02, target_strength=5.0):
    if iterations<=0:
        return bb.astype(np.float32)
    x=torch.tensor(bb,dtype=torch.float32).unsqueeze(0)
    p=torch.nn.Parameter(x.clone())
    opt=torch.optim.Adam([p],lr=lr)
    target=None
    if target_ca is not None and target_strength>0:
        target_np=np.asarray(target_ca,dtype=np.float32)
        if len(target_np)!=bb.shape[0]:
            target_np=resample_polyline(target_np,bb.shape[0])
        target=torch.tensor(target_np,dtype=torch.float32).unsqueeze(0)
    for _ in range(int(iterations)):
        opt.zero_grad()
        losses=combined_backbone_loss(p,None,w_bond=200.0,w_angle=80.0,w_omega=80.0,w_rama=5.0,w_clash=20.0)
        total=losses['total']
        if target is not None:
            total=total+target_strength*((p[:,:,1,:]-target)**2).sum(-1).mean()
        total.backward(); opt.step()
    return p.detach()[0].cpu().numpy().astype(np.float32)


def native_similarity_metrics(bb,native_info,contact_cutoff=10.0,overlap_cutoff=8.0,align_target='ligand'):
    if native_info is None:
        return {'native_score':0.0,'native_contact_frac':0.0,'native_overlap':0.0,'native_covered':0.0,'native_centroid_dist':999.0,'native_rg_ratio':0.0,'native_rmsd':999.0}
    CA=bb[:,1,:]
    rec=native_info['rec_ca']; iface=native_info['iface_ca']
    target=native_ca_target(native_info,len(CA),align_target)
    d_rec=np.linalg.norm(CA[:,None,:]-rec[None,:,:],axis=-1).min(1)
    contact=float((d_rec<=contact_cutoff).mean())
    d_gn=np.linalg.norm(CA[:,None,:]-iface[None,:,:],axis=-1).min(1)
    overlap=float((d_gn<=overlap_cutoff).mean())
    d_ng=np.linalg.norm(iface[:,None,:]-CA[None,:,:],axis=-1).min(1)
    covered=float((d_ng<=overlap_cutoff).mean())
    cdist=float(np.linalg.norm(CA.mean(0)-target.mean(0)))
    rg=float(np.sqrt(((CA-CA.mean(0,keepdims=True))**2).sum(1).mean()))
    nrg=float(np.sqrt(((target-target.mean(0,keepdims=True))**2).sum(1).mean()))
    ratio=rg/(nrg+1e-8)
    R,src_c,tgt_c=kabsch(CA,target)
    aligned=(CA-src_c[None,:])@R.T+tgt_c[None,:]
    rmsd=float(np.sqrt(((aligned-target)**2).sum(1).mean()))
    score=0.25*contact+0.25*overlap+0.15*covered+0.15*np.exp(-cdist/10.0)+0.20*np.exp(-rmsd/5.0)-0.10*abs(np.log(max(ratio,1e-6)))
    return {'native_score':float(score),'native_contact_frac':contact,'native_overlap':overlap,'native_covered':covered,'native_centroid_dist':cdist,'native_rg_ratio':float(ratio),'native_rmsd':rmsd}





def place_backbone_hotspot_guided(bb, surf_np, pocket_info, cand_idx, contact_offset=6.0, jitter=2.0):
    bb=bb.copy()
    centers=pocket_info.get('hotspot_centers', None)
    if centers is None or len(centers)==0:
        return place_backbone_predicted_pocket(bb,surf_np,pocket_info,cand_idx,contact_offset,jitter)
    idx=cand_idx % min(len(centers),16)
    anchor=np.asarray(centers[idx],dtype=np.float32)
    surf_center=surf_np.mean(0)
    normal=anchor-surf_center
    normal=normal/(np.linalg.norm(normal)+1e-8)
    R=random_rotation(cand_idx+211, dtype=bb.dtype)
    ca=bb[:,1,:]
    bb=((bb.reshape(-1,3)-ca.mean(0,keepdims=True))@R.T).reshape(bb.shape)
    ca=bb[:,1,:]
    interface_idx=len(ca)//2
    rng=np.random.default_rng(24680+cand_idx*43)
    target=anchor+normal*contact_offset
    if jitter>0:
        target=target+rng.normal(0.0,jitter,size=3).astype(np.float32)
    bb=bb+(target-ca[interface_idx])[None,None,:]
    return bb.astype(np.float32)


def receptor_hotspot_score(CA, pocket_info, min_frac=0.35, near_cutoff=10.0):
    if pocket_info is None:
        return 0.0, 0.0, 999.0
    centers=pocket_info.get('hotspot_centers', None)
    if centers is None or len(centers)==0:
        center=np.asarray(pocket_info.get('center', CA.mean(0)), dtype=np.float32).reshape(1,3)
    else:
        center=np.asarray(centers[:min(8,len(centers))], dtype=np.float32)
    hot_center=center.mean(0)
    d_hot=np.linalg.norm(CA[:,None,:]-center[None,:,:],axis=-1).min(1)
    hot_frac=float((d_hot<=near_cutoff).mean())
    centroid_dist=float(np.linalg.norm(CA.mean(0)-hot_center))
    penalty=max(0.0, min_frac-hot_frac)*25.0 + min(centroid_dist,40.0)/8.0
    return penalty, hot_frac, centroid_dist


def score(idx, topology, bb, ss, surf_np, pdb, pocket_info=None, w_stage3_pocket=5.0, native_info=None, w_native=0.0, native_contact_cutoff=10.0, native_overlap_cutoff=8.0, native_align_target='ligand'):







    r=validate_backbone(bb); CA=bb[:,1,:]; L=len(CA)







    rg=float(np.sqrt(((CA-CA.mean(0,keepdims=True))**2).sum(-1).mean()))







    d=np.linalg.norm(CA[:,None,:]-surf_np[None,:,:],axis=-1).min(1)







    contact=float(((d>=3.5)&(d<=10.0)).mean()); too_close=float((d<3.0).mean())







    adj_ca=np.linalg.norm(CA[1:]-CA[:-1],axis=1) if L>1 else np.array([3.8])
    adj_pen=float(np.maximum(0.0,3.35-adj_ca).sum()+np.maximum(0.0,adj_ca-4.25).sum())*20.0
    geom=abs(r['ca_ca_mean']-3.8)*10+r['ca_ca_std']*5+adj_pen







    idx_arr=np.arange(L); sep=np.abs(idx_arr[:,None]-idx_arr[None,:])



    d_ca=np.linalg.norm(CA[:,None,:]-CA[None,:,:],axis=-1)



    soft_clash=float(np.maximum(0.0, 3.6-d_ca[sep>=3]).sum())



    clash=r['n_clashes']*10+too_close*50+soft_clash*5







    target_rg=max(6.0,2.2*(L**(1/3))); comp=((rg-target_rg)/target_rg)**2*10







    cont=max(0,0.35-contact)*20



    native=native_similarity_metrics(bb,native_info,native_contact_cutoff,native_overlap_cutoff,native_align_target)
    hotspot_pen=0.0
    hot_frac=0.0
    hot_centroid_dist=999.0
    if pocket_info is not None:
        hotspot_pen, hot_frac, hot_centroid_dist = receptor_hotspot_score(CA, pocket_info, min_frac=0.42, near_cutoff=9.0)



    total=geom+clash+comp+cont+hotspot_pen-w_native*native['native_score']



    return dict(idx=idx,topology=topology,total_score=total,geom_score=geom,clash_score=clash,







        compact_score=comp,contact_score=cont,ca_ca_mean=r['ca_ca_mean'],ca_ca_std=r['ca_ca_std'],







        n_clashes=r['n_clashes'],rg=rg,contact_frac=contact,hotspot_frac=hot_frac,hotspot_centroid_dist=hot_centroid_dist,hotspot_score=hotspot_pen,output_pdb=pdb,**native)























def run_mpnn(mpnn_dir, pdb_path, out_dir, nseq, temp, ca_only):
    script=os.path.join(mpnn_dir,'protein_mpnn_run.py')
    os.makedirs(out_dir,exist_ok=True)
    if os.path.isdir(pdb_path):
        pdbs=[os.path.join(pdb_path,x) for x in sorted(os.listdir(pdb_path)) if x.endswith('.pdb')]
    else:
        pdbs=[pdb_path]
    if not pdbs:
        raise FileNotFoundError(f'No PDB files found for ProteinMPNN input: {pdb_path}')
    for pdb in pdbs:
        target_out=os.path.join(out_dir,os.path.splitext(os.path.basename(pdb))[0]) if len(pdbs)>1 else out_dir
        os.makedirs(target_out,exist_ok=True)
        cmd=[sys.executable,script,'--pdb_path',pdb,'--out_folder',target_out,'--num_seq_per_target',str(nseq),'--sampling_temp',str(temp),'--batch_size','1']
        if ca_only:
            cmd.append('--ca_only')
        subprocess.run(cmd,cwd=mpnn_dir,check=True)


def main():







    ap=argparse.ArgumentParser()







    ap.add_argument('--rec_npz',required=True); ap.add_argument('--stage4_ckpt',required=True); ap.add_argument('--stage3_ckpt',default='')







    ap.add_argument('--out_dir',required=True); ap.add_argument('--num_candidates',type=int,default=50); ap.add_argument('--top_k',type=int,default=10)







    ap.add_argument('--num_residues',type=int,default=48); ap.add_argument('--max_residues',type=int,default=128)







    ap.add_argument('--topology',default='three_helix_bundle'); ap.add_argument('--topology_mix',default='')







    ap.add_argument('--noise_deg',type=float,default=10.0); ap.add_argument('--generator_blend',type=float,default=0.15)







    ap.add_argument('--refine_steps',type=int,default=100); ap.add_argument('--refine_lr',type=float,default=0.01); ap.add_argument('--seed',type=int,default=2026)







    ap.add_argument('--K',type=int,default=32); ap.add_argument('--seq_len',type=int,default=512); ap.add_argument('--d_model',type=int,default=256); ap.add_argument('--nhead',type=int,default=8); ap.add_argument('--nlayers_surf',type=int,default=6); ap.add_argument('--nlayers_gen',type=int,default=4)







    ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')







    ap.add_argument('--w_bond',type=float,default=100.); ap.add_argument('--w_angle',type=float,default=50.); ap.add_argument('--w_omega',type=float,default=80.); ap.add_argument('--w_rama',type=float,default=10.); ap.add_argument('--w_clash',type=float,default=20.)







    ap.add_argument('--w_ss',type=float,default=80.); ap.add_argument('--w_rep',type=float,default=100.); ap.add_argument('--w_compact',type=float,default=2.); ap.add_argument('--w_pack',type=float,default=20.); ap.add_argument('--w_contact',type=float,default=5.)







    ap.add_argument('--repulsion_dist',type=float,default=3.4); ap.add_argument('--target_rg',type=float,default=0.); ap.add_argument('--min_long_contacts',type=int,default=2); ap.add_argument('--long_contact_dist',type=float,default=8.)







    ap.add_argument('--target_contact_frac',type=float,default=0.35); ap.add_argument('--surface_min_dist',type=float,default=3.5); ap.add_argument('--surface_max_dist',type=float,default=10.); ap.add_argument('--placement_offset',type=float,default=6.0); ap.add_argument('--pocket_source',choices=['stage4','stage3','hotspot','surface'],default='stage4'); ap.add_argument('--pocket_jitter',type=float,default=2.0); ap.add_argument('--w_stage3_pocket',type=float,default=5.0); ap.add_argument('--pack_target_dist',type=float,default=10.0); ap.add_argument('--pack_min_dist',type=float,default=6.0); ap.add_argument('--pack_max_dist',type=float,default=13.0); ap.add_argument('--pack_axis_abs_cos',type=float,default=0.65)







    ap.add_argument('--generator_mode',choices=['torsion','parametric_bundle'],default='torsion'); ap.add_argument('--topology_family',choices=['model','auto','helical_peptide','helix_loop_helix','three_helix_bundle','four_helix_bundle','beta_hairpin','coil_peptide'],default='model'); ap.add_argument('--topology_exploration_frac',type=float,default=0.35); ap.add_argument('--native_pdb',default=''); ap.add_argument('--native_receptor_chains',default=''); ap.add_argument('--native_ligand_chains',default=''); ap.add_argument('--native_guided',action='store_true'); ap.add_argument('--native_use_ligand_length',action='store_true'); ap.add_argument('--native_jitter',type=float,default=1.0); ap.add_argument('--native_align_target',choices=['ligand','interface'],default='ligand'); ap.add_argument('--native_reverse',choices=['auto','both','yes','no'],default='auto'); ap.add_argument('--native_trace_strength',type=float,default=0.0,help='After native-guided placement, smoothly project CA trace toward native target; 0 disables, 1 follows target trace'); ap.add_argument('--native_trace_smooth',type=int,default=2); ap.add_argument('--native_relax_steps',type=int,default=0); ap.add_argument('--native_relax_lr',type=float,default=0.02); ap.add_argument('--native_relax_strength',type=float,default=5.0); ap.add_argument('--w_native',type=float,default=0.0); ap.add_argument('--w_native_shape',type=float,default=0.0); ap.add_argument('--native_contact_cutoff',type=float,default=10.0); ap.add_argument('--native_overlap_cutoff',type=float,default=8.0); ap.add_argument('--template_pdb',default='',help='Debug/scaffold-based mode only; not de novo'); ap.add_argument('--template_start',type=int,default=1); ap.add_argument('--template_chain',default=''); ap.add_argument('--mirror_output',choices=['none','x','y','z'],default='none'); ap.add_argument('--run_proteinmpnn',action='store_true'); ap.add_argument('--proteinmpnn_dir',default='/data2/jiangjiaqi/srzhang/InversionDock/Code/protein/ProteinMPNN'); ap.add_argument('--mpnn_num_seq_per_target',type=int,default=4); ap.add_argument('--mpnn_sampling_temp',default='0.1'); ap.add_argument('--mpnn_ca_only',action='store_true')







    args=ap.parse_args(); set_seed(args.seed); device=torch.device(args.device)







    os.makedirs(args.out_dir,exist_ok=True); cand_dir=os.path.join(args.out_dir,'candidates'); top_dir=os.path.join(args.out_dir,'top'); os.makedirs(cand_dir,exist_ok=True); os.makedirs(top_dir,exist_ok=True)

    if args.native_use_ligand_length and args.native_pdb and args.native_ligand_chains:
        native_lig_len=len(read_pdb_ca_chains(args.native_pdb,args.native_ligand_chains))
        if native_lig_len > 0 and native_lig_len != args.num_residues:
            print(f'[Native guidance] overriding num_residues {args.num_residues} -> native ligand length {native_lig_len}')
            args.num_residues=native_lig_len

    rf,rc,rp,surf_np=load_receptor_npz(args.rec_npz,args.K,args.seq_len,device); surf=torch.from_numpy(surf_np).to(device); model=build_generator(args,device); stage3_model=build_stage3_guidance(args,device); native_info=build_native_guidance(args)

    if args.pocket_source in ('stage4','hotspot'):
        pocket_info=predict_stage4_pocket(model,rf,rc,rp,args.num_residues,surf_np)
    elif args.pocket_source=='stage3':
        pocket_info=predict_stage3_pocket(stage3_model,rf,rc,rp,surf_np)
    else:
        pocket_info={'center':surf_np.mean(0).astype(np.float32),'radius':float(np.sqrt(((surf_np-surf_np.mean(0)[None,:])**2).sum(1).mean())),'source':'surface','confidence':0.0}







    tops=[x.strip() for x in args.topology_mix.split(',') if x.strip()] or [args.topology]







    with torch.no_grad():
        model_out=model(rf,rc,rp,args.num_residues)
        base=normalize_torsion_sincos(model_out[0])
        model_ss_logits=model_out[1]
        model_topology_logits=model_out[4]







    rows=[]







    for i in range(args.num_candidates):







        if args.topology_family == 'model':
            model_top_idx=int(torch.argmax(model_topology_logits, dim=-1).item())
            top=TOPOLOGY_FAMILIES[model_top_idx]
            if i > 0 and i < len(TOPOLOGY_FAMILIES) and i % max(2, int(round(1.0/max(args.topology_exploration_frac,1e-3)))) == 0:
                top=TOPOLOGY_FAMILIES[i % len(TOPOLOGY_FAMILIES)]
        else:
            top=choose_auto_topology(pocket_info,surf_np,i,args.topology_exploration_frac) if args.topology_family=='auto' else args.topology_family
        if args.generator_mode!='parametric_bundle' and args.topology_family not in ('auto','model'): top=tops[i%len(tops)]
        if args.topology_family == 'model' and args.generator_mode != 'parametric_bundle':
            ss=topology_ss(top,args.num_residues,device)
            if top == 'coil_peptide':
                ss = topology_ss('coil_peptide', args.num_residues, device)
        else:
            ss=topology_ss(top,args.num_residues,device)
        if args.template_pdb:
            print('[WARN] --template_pdb uses scaffold-based debug mode, not de novo') if i==0 else None
            bb=load_template_backbone(args.template_pdb,args.template_start,args.num_residues,args.template_chain or None)
            R=random_rotation(i+args.seed, dtype=bb.dtype); bb=((bb.reshape(-1,3))@R.T).reshape(bb.shape)
        elif args.generator_mode=='parametric_bundle':
            if top == 'coil_peptide':
                top = 'helical_peptide'
            bb=build_parametric_bundle(args.num_residues, family=top, cand_idx=i, seed=args.seed)
            ss = topology_ss(top, args.num_residues, device)
        else:
            x=sample_torsion(base,ss,args.noise_deg,args.generator_blend); bb=refine(x,ss,surf,args,native_info)[0].cpu().numpy()
        if args.native_guided and native_info is not None:
            bb,target_ca=place_backbone_native_guided(bb,native_info,i,args.native_jitter,args.native_align_target,args.native_reverse)
            bb=project_backbone_to_native_trace(bb,target_ca,args.native_trace_strength,args.native_trace_smooth)
            bb=relax_projected_backbone(bb,target_ca,args.native_relax_steps,args.native_relax_lr,args.native_relax_strength)
        else:
            bb=(place_backbone_hotspot_guided(bb,surf_np,pocket_info,i,args.placement_offset,args.pocket_jitter) if args.pocket_source=='hotspot' else (place_backbone_predicted_pocket(bb,surf_np,pocket_info,i,args.placement_offset,args.pocket_jitter) if args.pocket_source in ('stage4','stage3') else place_backbone_near_surface(bb,surf_np,i,args.placement_offset)))
        if not (args.native_guided and native_info is not None):
            bb=maybe_mirror_backbone(bb,args.mirror_output)
        ss_np=ss[0].cpu().numpy()







        pdb=os.path.join(cand_dir,f'cand_{i:04d}_{top}.pdb'); write_pdb(bb,pdb,residue_name='ALA',ss_labels=ss_np); row=score(i,top,bb,ss_np,surf_np,pdb,pocket_info,args.w_stage3_pocket,native_info,args.w_native,args.native_contact_cutoff,args.native_overlap_cutoff,args.native_align_target); rows.append(row)







        print(f"[Cand {i:04d}] {top} score={row['total_score']:.3f} clash={row['n_clashes']} caca={row['ca_ca_mean']:.3f} rg={row['rg']:.2f} contact={row['contact_frac']:.2f} native={row.get('native_score',0):.2f} rmsd={row.get('native_rmsd',999):.2f} overlap={row.get('native_overlap',0):.2f}")







    rows=sorted(rows,key=lambda r:r['total_score'])







    for old in os.listdir(top_dir):
        old_path=os.path.join(top_dir, old)
        if os.path.isfile(old_path):
            os.remove(old_path)

    for rank,row in enumerate(rows[:args.top_k]):







        dst=os.path.join(top_dir,f"top_{rank:03d}_score_{row['total_score']:.3f}_"+os.path.basename(row['output_pdb'])); shutil.copyfile(row['output_pdb'],dst); row['output_pdb']=dst







    with open(os.path.join(args.out_dir,'scores.csv'),'w',newline='') as f:







        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)







    print('[Done]',args.out_dir)







    if args.run_proteinmpnn: run_mpnn(args.proteinmpnn_dir,top_dir,os.path.join(args.out_dir,'proteinmpnn'),args.mpnn_num_seq_per_target,args.mpnn_sampling_temp,args.mpnn_ca_only)















if __name__=='__main__': main()







