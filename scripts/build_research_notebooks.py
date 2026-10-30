"""Create the presentation-oriented, artifact-backed FNEWS notebook suite."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/research"

BOOT = '''from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path.cwd()
if not (ROOT / "data/research_dataset").exists():
    ROOT = Path("/Users/aka_ank/Desktop/FNEWS")
DATA = ROOT / "data/research_dataset"
REPORTS = ROOT / "reports"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.titlesize": 13, "axes.labelsize": 11})

def paired(split):
    frame = pd.read_parquet(DATA / "manifests" / f"{split}_manifest.parquet")
    return frame.loc[(frame.source_dataset == "Fakeddit") & (frame.download_status == "verified_reused_or_processed")].copy().reset_index(drop=True)

def canonical_source_labels(raw):
    """Fakeddit raw 1=true; return 0=SOURCE_TRUE, 1=SOURCE_NON_TRUE."""
    return 1 - np.asarray(raw, dtype=int)

def source_names(values):
    return np.where(np.asarray(values) == 0, "SOURCE_TRUE", "SOURCE_NON_TRUE")

def show_frame(frame, rows=10):
    display(frame.head(rows))
'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def write(name: str, title: str, objective: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb["cells"] = [
        md(f"# {title}\n\n**Research objective.** {objective}\n\n"
           "All results are read from local FNEWS artifacts. `SOURCE_TRUE` and `SOURCE_NON_TRUE` are Fakeddit source-taxonomy labels, not independently verified factuality labels. The held-out test partition is not opened by these notebooks except where a prior frozen metric is explicitly quoted."),
        code(BOOT),
        *cells,
    ]
    nbf.write(nb, OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write("01_dataset_eda.ipynb", "FNEWS dataset EDA", "Verify the bounded paired Fakeddit dataset, its source labels, modality coverage, and known limitations.", [
        md("## Dataset and provenance\n\nFakeddit is a Reddit-derived multimodal benchmark. The local research dataset contains only validated text-image pairs. Its binary source label is derived from a six-way taxonomy: raw `1` is source-true; raw `0` aggregates non-true categories. This makes it useful for controlled benchmarking but not a general factuality corpus."),
        code("splits={s:paired(s) for s in ['train','validate','test']}\nsummary=pd.DataFrame([{'split':s,'pairs':len(f),'raw_0_non_true':int((f.fake_news_label==0).sum()),'raw_1_true':int((f.fake_news_label==1).sum())} for s,f in splits.items()])\nshow_frame(summary)\nprint('Verified paired total:',sum(summary.pairs))\nprint('Processed dataset GiB:',sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())/1024**3)"),
        code("all_pairs=pd.concat(splits.values(),ignore_index=True); all_pairs['canonical']=source_names(canonical_source_labels(all_pairs.fake_news_label))\nax=all_pairs.canonical.value_counts().reindex(['SOURCE_TRUE','SOURCE_NON_TRUE']).plot.bar(color=['#3f7f61','#b86753'],title='Paired records by source taxonomy');ax.set_ylabel('Records');plt.show()\nall_pairs['chars']=all_pairs.text.str.len();all_pairs['words']=all_pairs.text.str.split().str.len();display(all_pairs.groupby('canonical')[['chars','words']].describe().round(2))"),
        md("## Text EDA\n\nTF-IDF uses the cleaned title text. The distributions below describe the actual paired dataset, not an inferred web-news distribution."),
        code("fig,axes=plt.subplots(1,2,figsize=(12,4));\nfor label,group in all_pairs.groupby('canonical'):\n    axes[0].hist(group.chars,bins=35,alpha=.55,label=label);axes[1].hist(group.words,bins=25,alpha=.55,label=label)\naxes[0].set(title='Character count',xlabel='Characters');axes[1].set(title='Word count',xlabel='Words');[a.legend() for a in axes];plt.show()\nprint('Duplicate-title members:',int(all_pairs.duplicated('text',keep=False).sum()))"),
        md("## Image and multimodal EDA\n\nImages were decoded, bounded, and checked for cross-split duplicate risk during construction. The plot and gallery use only locally verified image files."),
        code("all_pairs['aspect']=all_pairs.image_width/all_pairs.image_height;all_pairs['pixels']=all_pairs.image_width*all_pairs.image_height\nfig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].hist(all_pairs.aspect,bins=35,color='#3f7f61');axes[0].set(title='Aspect ratio',xlabel='width / height');axes[1].hist(all_pairs.pixels,bins=35,color='#476f99');axes[1].set(title='Processed image pixels',xlabel='pixels');plt.show()\nprint('Missing text:',int(all_pairs.text.isna().sum()),'Missing image path:',int(all_pairs.image_path.isna().sum()),'Unique image hashes:',all_pairs.image_hash.nunique())"),
        code("gallery=all_pairs.groupby('canonical',group_keys=False).sample(3,random_state=42);fig,axes=plt.subplots(2,3,figsize=(11,6));\nfor ax,(_,row) in zip(axes.flat,gallery.iterrows()):\n    ax.imshow(Image.open(DATA/row.image_path).convert('RGB'));ax.set_title(row.canonical+'\\n'+row.text[:45]);ax.axis('off')\nplt.tight_layout();plt.show()"),
        md("## Findings and limitations\n\nThe observed dataset has 4,233 verified pairs in preserved train/validation/test membership. Titles can be short or context-poor, source categories are heterogeneous, and text/image pairing does not independently prove claim truth. Recommended preprocessing: fit statistics on train only, preserve source labels, keep images RGB-decoded, and do not report live-news factuality accuracy from this benchmark."),
    ])
    write("02_preprocessing.ipynb", "Preprocessing and leakage prevention", "Demonstrate the exact saved text and image preprocessing boundaries.", [
        md("## Train-only preprocessing principle\n\nVocabulary, inverse-document-frequency statistics, calibration parameters, and model selection must be fitted on training data only. Validation is used for recorded selection/audit tasks; the frozen test result is never reused here."),
        code("train,val=paired('train'),paired('validate');model=joblib.load(REPORTS/'experiments/research_dataset_baselines/tfidf_svm/model.joblib');tfidf=model.named_steps['tfidf'];svm=model.named_steps['svm'];sample=train.iloc[0];valid=val.iloc[0]\nprint('Train sample:',sample.text);print('Validation sample:',valid.text);print('SVM classes:',svm.classes_.tolist());print('TF-IDF configuration:',tfidf.get_params())\nprint('Raw Fakeddit 1 = SOURCE_TRUE; raw 0 = SOURCE_NON_TRUE')"),
        code("x_train=tfidf.transform([sample.text]);x_val=tfidf.transform([valid.text]);print('Train transform shape:',x_train.shape,'nonzero:',x_train.nnz);print('Validation transform shape:',x_val.shape,'nonzero:',x_val.nnz);print('No refit is performed on validation.')"),
        md("## Text processing\n\nThe saved vectorizer lowercases and produces unigram/bigram sparse vectors. Stopword removal is deliberately not introduced because short titles can lose useful grammatical context. Whitespace cleanup occurs upstream in `clean_title`."),
        code("from torchvision.models import ResNet18_Weights\ntransform=ResNet18_Weights.DEFAULT.transforms();image=Image.open(DATA/valid.image_path).convert('RGB');tensor=transform(image);print('Original image:',image.size,'RGB');print('ResNet input tensor:',tuple(tensor.shape),tensor.dtype)\nprint('Image path:',valid.image_path)"),
        md("## Leakage controls\n\nOfficial split membership is preserved. Exact hashes and perceptual hashes were used to exclude cross-split duplicate or near-duplicate images. The application validates uploaded image bytes before persistence; URL extraction applies SSRF, redirect, robots, content-type, and size controls."),
    ])
    write("03_feature_extraction.ipynb", "Feature extraction", "Explain and inspect actual sparse, image, and frozen-embedding feature configurations.", [
        md("## TF-IDF sparse features\n\nFor document *d* and term *t*, TF-IDF combines term frequency with inverse document frequency. The saved SVM vectorizer uses at most 50,000 unigram/bigram terms, L2 normalization, `min_df=2`, and sublinear term frequency."),
        code("val=paired('validate');svm_pipe=joblib.load(REPORTS/'experiments/research_dataset_baselines/tfidf_svm/model.joblib');vec=svm_pipe.named_steps['tfidf'];matrix=vec.transform(val.text);print('Validation sparse matrix:',matrix.shape,'density:',matrix.nnz/(matrix.shape[0]*matrix.shape[1]));print('Vocabulary features:',len(vec.get_feature_names_out()))"),
        code("coef=svm_pipe.named_steps['svm'].coef_[0];terms=vec.get_feature_names_out();\nfor title,idx in [('Raw class 1 / SOURCE_TRUE',coef.argsort()[-15:][::-1]),('Raw class 0 / SOURCE_NON_TRUE',coef.argsort()[:15])]:\n print('\\n'+title);display(pd.DataFrame({'term':terms[idx],'weight':coef[idx]}))"),
        md("## Frozen text, image, and CLIP representations\n\nThe saved pretrained-text baseline is a frozen encoder baseline; its stored validation predictions are used rather than recomputing transformer embeddings. ResNet-18 uses ImageNet preprocessing and a two-class head. CLIP ViT-B/32 contributes 512-dimensional image and text embeddings to the archived gated head."),
        code("checkpoint_path=REPORTS/'multimodal_fusion/clip_fusion_v1/gated/best_model.pt'\nimport torch\ncheckpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False);print('CLIP fusion checkpoint config:',checkpoint['config']);print('Stored feature mean shape:',tuple(checkpoint['feature_mean'].shape));print('Stored feature std shape:',tuple(checkpoint['feature_std'].shape));print('Expected concatenation: image 512 + text 512 = 1024')"),
        code("from torchvision.models import ResNet18_Weights\nrow=val.iloc[0];pixels=ResNet18_Weights.DEFAULT.transforms()(Image.open(DATA/row.image_path).convert('RGB'));print('Real validation ResNet tensor:',tuple(pixels.unsqueeze(0).shape));print('No supervised dimensionality reduction is fitted on validation labels.')"),
        md("## Reproduction note\n\nEmbedding caches are not required for this notebook. Shapes are read from the preserved checkpoint and an actual bounded validation input is transformed with the recorded image pipeline."),
    ])
    write("04_text_model_training.ipynb", "Text model training evidence", "Document saved TF-IDF + LinearSVC and frozen pretrained-text baseline evidence without inventing curves.", [
        md("## TF-IDF + LinearSVC\n\nLinearSVC learns a separating margin. Its decision function is not a calibrated probability. The stored classifier was fit with raw Fakeddit labels, where class 1 means source-true; the application translates this at presentation time."),
        code("metrics=json.loads((REPORTS/'experiments/research_dataset_baselines/metrics.json').read_text());display(pd.DataFrame(metrics['results']['tfidf_svm']['validation']['per_class']).T);print('Saved validation accuracy:',metrics['results']['tfidf_svm']['validation']['accuracy']);print('Saved validation macro-F1:',metrics['results']['tfidf_svm']['validation']['macro_f1'])\npipe=joblib.load(REPORTS/'experiments/research_dataset_baselines/tfidf_svm/model.joblib');print(pipe)"),
        code("pred=pd.read_parquet(REPORTS/'experiments/research_dataset_baselines/tfidf_svm/validation_predictions.parquet');print('Stored prediction records:',len(pred));display(pred.head())\nbert=pd.read_parquet(REPORTS/'experiments/research_dataset_baselines/pretrained_text_frozen/validation_predictions.parquet');print('Frozen pretrained-text records:',len(bert))"),
        md("## Training methodology and limitations\n\nThe SVM has no epoch-level history; plotting a learning curve or loss curve would be fabricated. Its regularization and feature configuration are recovered from the saved pipeline. The frozen pretrained-text baseline provides a comparison but does not establish domain-general factuality performance."),
    ])
    write("05_image_model_training.ipynb", "Image model training evidence", "Document the preserved ResNet-18 transfer-learning artifact and its limitations.", [
        md("## ResNet-18 transfer learning\n\nThe image model uses ImageNet-pretrained ResNet-18 preprocessing and a two-class classification head. Image-only prediction can recognize source-correlated visual patterns but cannot verify a claim’s factual status."),
        code("history=pd.read_csv(REPORTS/'experiments/research_dataset_baselines/resnet18_frozen/history.csv');display(history.head());print('History columns:',history.columns.tolist())\nax=history.plot(x=history.columns[0],y=[c for c in history.columns if c!=history.columns[0]],marker='o',title='Saved ResNet-18 training history');ax.set_xlabel('Recorded training step');plt.show()"),
        code("pred=pd.read_parquet(REPORTS/'experiments/research_dataset_baselines/resnet18_frozen/validation_predictions.parquet');y=canonical_source_labels(pred.label);p=canonical_source_labels(pred.prediction);from sklearn.metrics import confusion_matrix,f1_score\ncm=confusion_matrix(y,p,labels=[0,1]);print('Canonical [SOURCE_TRUE,SOURCE_NON_TRUE] matrix:',cm.tolist());print('Macro-F1:',f1_score(y,p,average='macro'));display(pd.DataFrame(cm,index=['SOURCE_TRUE','SOURCE_NON_TRUE'],columns=['SOURCE_TRUE','SOURCE_NON_TRUE']))"),
        md("## Interpretation\n\nThe saved 933-pair validation audit has source-true recall 66.84%, source-non-true recall 71.04%, and macro-F1 68.61%. These are source-taxonomy results, not claim-factuality results. Use image signals only alongside explicit task boundaries."),
    ])
    write("06_multimodal_fusion.ipynb", "Multimodal fusion", "Compare saved weighted late fusion with the archived CLIP gated-fusion checkpoint.", [
        md("## Weighted late fusion\n\n`P_non_true = alpha × P_text_non_true + (1-alpha) × P_image_non_true`. The saved alpha is 0.41 for text and 0.59 for image. Alpha was selected on validation, so the displayed validation result is selection-set performance, not independent confirmation."),
        code("weighted=pd.read_parquet(REPORTS/'weighted_late_fusion_v1/validation_predictions.parquet');alpha=0.41;text_non=1-weighted.text_probability_fake;image_non=1-weighted.image_probability_fake;combined=alpha*text_non+(1-alpha)*image_non;print('Maximum reconstruction error:',float(np.abs(combined-(1-weighted.combined_probability_fake)).max()));print('All stored raw-label decisions reproduced:',bool(np.array_equal((combined<.5).astype(int),weighted.prediction.to_numpy())))\nmetrics=json.loads((REPORTS/'weighted_late_fusion_v1/metrics.json').read_text());display(pd.DataFrame(metrics['validation']).T)"),
        md("## CLIP gated feature fusion\n\nThe archived head combines frozen 512-D CLIP image and text features. A learned sigmoid gate is feature-wise, not scalar: `g = sigmoid(W[z_text;z_image]+b)` and `z_fused = g*z_text + (1-g)*z_image`. This differs from weighted late fusion, which combines final class probabilities; neither is cross-attention."),
        code("checkpoint=torch.load(REPORTS/'multimodal_fusion/clip_fusion_v1/gated/best_model.pt',map_location='cpu',weights_only=False);print('Gated configuration:',checkpoint['config']);print('Feature statistics shape:',tuple(checkpoint['feature_mean'].shape));gated=pd.read_parquet(REPORTS/'multimodal_fusion/clip_fusion_v1/gated/validation_predictions.parquet');print('Gated validation records:',len(gated))"),
        code("from sklearn.metrics import confusion_matrix,f1_score\ny=canonical_source_labels(gated.label);p=canonical_source_labels(gated.prediction);cm=confusion_matrix(y,p,labels=[0,1]);print(cm.tolist(),'macro-F1',f1_score(y,p,average='macro'));display(pd.DataFrame(cm,index=['SOURCE_TRUE','SOURCE_NON_TRUE'],columns=['SOURCE_TRUE','SOURCE_NON_TRUE']))"),
        md("## Application status\n\nThe gated model is preserved but archived/inactive in the application. It must not be silently substituted for the active text-only policy."),
    ])
    write("07_model_evaluation.ipynb", "Model evaluation", "Compare saved, matched validation predictions using correct source-taxonomy semantics.", [
        md("## Evaluation protocol\n\nMacro-F1 weights both classes equally, which is valuable when source-taxonomy classes differ in frequency. Accuracy alone can hide asymmetric errors. All comparisons below use the same 933 validation IDs and are not external factuality scores."),
        code("from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,f1_score\npaths={'LinearSVC':REPORTS/'experiments/research_dataset_baselines/tfidf_svm/validation_predictions.parquet','Frozen pretrained text':REPORTS/'experiments/research_dataset_baselines/pretrained_text_frozen/validation_predictions.parquet','ResNet-18':REPORTS/'experiments/research_dataset_baselines/resnet18_frozen/validation_predictions.parquet','Weighted fusion':REPORTS/'weighted_late_fusion_v1/validation_predictions.parquet','CLIP gated (archived)':REPORTS/'multimodal_fusion/clip_fusion_v1/gated/validation_predictions.parquet'}\nrows=[]\nfor name,path in paths.items():\n d=pd.read_parquet(path);y=canonical_source_labels(d.label);p=canonical_source_labels(d.prediction);cm=confusion_matrix(y,p,labels=[0,1]);rows.append({'model':name,'ids':d.sample_id.nunique(),'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro'),'true_recall':cm[0,0]/cm[0].sum(),'non_true_recall':cm[1,1]/cm[1].sum(),'balanced_accuracy':np.mean([cm[0,0]/cm[0].sum(),cm[1,1]/cm[1].sum()]),'matrix':cm.tolist()})\ncomparison=pd.DataFrame(rows);display(comparison.sort_values('macro_f1',ascending=False))"),
        code("fig,ax=plt.subplots();comparison.set_index('model')[['macro_f1','balanced_accuracy']].sort_values('macro_f1').plot.barh(ax=ax,title='Matched validation comparison');ax.set_xlim(.5,1);plt.show()"),
        md("## Frozen test record\n\nA previously recorded single test result exists for the archived gated model (86.91% accuracy, 86.45% macro-F1 on 947 pairs). It is historical, was not reread here, and was not used for selection."),
    ])
    write("08_error_analysis.ipynb", "Error analysis", "Analyze saved source-true/source-non-true errors without relabeling source data.", [
        md("## Active SVM errors\n\nThe corrected canonical matrix is `[[287, 90], [171, 385]]` in `[SOURCE_TRUE, SOURCE_NON_TRUE]` order. Correct label presentation did not change these raw predictions."),
        code("fn=pd.read_csv(REPORTS/'root_cause/svm_source_non_true_false_negatives.csv');fp=pd.read_csv(REPORTS/'root_cause/false_positive_manifest.csv');print('SOURCE_NON_TRUE → SOURCE_TRUE:',len(fn));print('SOURCE_TRUE → SOURCE_NON_TRUE:',len(fp));display(fn.head(10))"),
        code("fig,axes=plt.subplots(1,2,figsize=(12,4));fn['6_way_label'].value_counts().sort_index().plot.bar(ax=axes[0],title='Non-true → true errors by six-way subtype');axes[0].set_xlabel('Six-way subtype');fn.text_length.plot.hist(ax=axes[1],bins=25,title='False-negative title length');axes[1].set_xlabel('Characters');plt.show()"),
        code("display(fn.groupby('subreddit').size().sort_values(ascending=False).head(12).rename('errors').to_frame());display(fn.sort_values('source_non_true_margin').head(12)[['text','subreddit','6_way_label','source_non_true_margin']])"),
        md("## Interpretation\n\nFailures include satire, misleading content, imposter content, false connection, and manipulation. Short titles and source-community style can limit lexical evidence. These labels are taxonomy categories, not universal fact-check outcomes; no record is relabeled in this notebook."),
    ])
    write("09_robustness_calibration.ipynb", "Calibration and robustness", "Present saved calibration, modality sensitivity, and raw-image robustness evidence with scope boundaries.", [
        md("## Calibration\n\nTemperature scaling maps logits to probabilities using a learned positive temperature. ECE measures calibration gap; Brier score measures squared probability error. Calibration does not establish factuality or repair domain shift."),
        code("cal=json.loads((REPORTS/'phase2/calibration/metrics.json').read_text());display(pd.json_normalize(cal));print('Saved temperature:',json.loads((REPORTS/'phase2/calibration/temperature.json').read_text()).get('temperature'))"),
        md("The saved assessment used 467 validation examples: ECE improved from 0.0889 to 0.0380 and Brier score from 0.1017 to 0.0898. This is a split-local calibration assessment."),
        code("ablation=json.loads((REPORTS/'phase2/modality_ablation/metrics.json').read_text());display(pd.DataFrame(ablation['conditions']).T)"),
        md("## Robustness\n\nFrozen modality shuffling changes macro-F1 from 87.07% paired to 70.77% with shuffled images and 63.82% with shuffled text. The raw-image sweep is a 32-pair, 19-condition subset experiment—not a full-validation robustness estimate. Severe blur sigma 3 changed subset macro-F1 from 87.50% to 77.03%."),
        code("paths=list((REPORTS/'phase2').rglob('*robust*.json'));print('Available robustness JSON:',[p.name for p in paths]);\nfor path in paths:\n print('\\n',path);display(pd.json_normalize(json.loads(path.read_text())).head())"),
    ])
    write("10_final_research_summary.ipynb", "Final FNEWS research summary", "Provide a presentation-ready overview grounded in verified local artifacts.", [
        md("## Problem, objective, and pipeline\n\nFNEWS studies source-taxonomy classification of paired Reddit text and images. The pipeline is: official metadata → verified bounded pairs → leakage controls → train-only preprocessing → saved baselines/fusion experiments → validation-only diagnostics → guarded local URL/text workflow. It is not a general-purpose factual-truth verifier."),
        md("## EDA totals at a glance\n\nThe dataset EDA confirms that this research version contains only verified text-image pairs. Fakeddit raw label `1` maps to `SOURCE_TRUE`; raw label `0` maps to the broader `SOURCE_NON_TRUE` taxonomy. The diagram below is an EDA summary, not a performance chart."),
        code("splits={s:paired(s) for s in ['train','validate','test']}\neda=pd.DataFrame([{'split':s.title(),'pairs':len(frame),'SOURCE_TRUE':int((frame.fake_news_label==1).sum()),'SOURCE_NON_TRUE':int((frame.fake_news_label==0).sum())} for s,frame in splits.items()]);display(eda)\nprint('Total verified paired samples:',int(eda.pairs.sum()))\nprint('Total SOURCE_TRUE:',int(eda.SOURCE_TRUE.sum()),'| Total SOURCE_NON_TRUE:',int(eda.SOURCE_NON_TRUE.sum()))"),
        code("fig,axes=plt.subplots(1,2,figsize=(12,4));eda.set_index('split')['pairs'].plot.bar(ax=axes[0],color='#3f7f61',title='EDA: verified paired samples by split');axes[0].set_ylabel('Pairs')\neda.set_index('split')[['SOURCE_TRUE','SOURCE_NON_TRUE']].plot.bar(stacked=True,ax=axes[1],color=['#4e8b6a','#b86753'],title='EDA: source taxonomy distribution');axes[1].set_ylabel('Pairs');plt.tight_layout();plt.show()"),
        md("## Training methodology\n\n**Text baseline.** The active TF-IDF + LinearSVC model learns a maximum-margin decision boundary from cleaned Fakeddit titles. The vectorizer is fitted on training text only with up to 50,000 unigram/bigram features, `min_df=2`, sublinear term frequency, and L2 normalization. A LinearSVC margin is not a calibrated probability.\n\n**Image baseline.** ResNet-18 uses ImageNet preprocessing and a two-class head on local images. Image-only results are limited because an image cannot independently establish the factual status of a title.\n\n**Fusion experiments.** Weighted late fusion combines compatible calibrated source-non-true probabilities with alpha 0.41/0.59. CLIP gated fusion combines frozen 512-D image and text embeddings with a learned feature-wise sigmoid gate. Both are retained research artifacts; the latter is archived/inactive in the website."),
        code("svm=joblib.load(REPORTS/'experiments/research_dataset_baselines/tfidf_svm/model.joblib');print('Saved TF-IDF + LinearSVC configuration:');print(svm)\nhistory=pd.read_csv(REPORTS/'experiments/research_dataset_baselines/resnet18_frozen/history.csv');print('ResNet history columns:',history.columns.tolist());print('Weighted late-fusion alpha:',json.loads((REPORTS/'weighted_late_fusion_v1/metrics.json').read_text())['alpha'])"),
        code("state=json.loads((ROOT/'PROJECT_STATE.json').read_text());display(pd.DataFrame(state['phase_status_checklist']))"),
        code("reliability=json.loads((REPORTS/'root_cause/classification_reliability_metrics.json').read_text());comparison=pd.DataFrame(reliability['model_metrics']).T;display(comparison[['accuracy','macro_f1','source_true_recall','source_non_true_recall','balanced_accuracy','matrix']])\ncomparison[['macro_f1','balanced_accuracy']].sort_values('macro_f1').plot.barh(title='Verified validation model comparison');plt.xlim(.5,1);plt.show()"),
        md("## Final findings\n\nThe archived CLIP gated model has the strongest saved matched validation result, but remains inactive. Weighted late fusion improves the active SVM on the selection validation set but has no independent post-selection validation. The active dashboard deliberately uses the verified text-only SVM and names its result SOURCE_TRUE/SOURCE_NON_TRUE.\n\n### Limitations and future work\n\nThe dataset’s taxonomy includes satire, misleading content, false connection, and manipulation; it is not independently verified live-news truth. External zero-shot evaluation and image-text mismatch detection remain user-deferred. Any future deployment decision requires legally usable external data, explicit labels, a frozen protocol, and an untouched evaluation split."),
        code("print('Checkpoint preservation:',state['phase_status_checklist'][1]['evidence']);print('No notebook in this suite trains a model or reads the held-out test manifest for selection.')"),
    ])


if __name__ == "__main__":
    main()
