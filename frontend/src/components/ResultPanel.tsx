import type { ClassificationDetail, AnalysisResponse } from '../types';

interface ResultPanelProps {
  classification: ClassificationDetail;
  localization?: AnalysisResponse['localization'];
  scanType: string;
  taskType?: string;
  analysisTimeMs: number;
}

export default function ResultPanel({ classification, localization, scanType, taskType = 'classification', analysisTimeMs }: ResultPanelProps) {
  const { all_scores, confidence, severity, top_label } = classification;
  const isDetection = taskType === 'detection';

  const scores = all_scores ? Object.entries(all_scores).sort(([, a], [, b]) => b - a) : [];
  const modelName = ({
    chest_xray: 'RAD-DINO · 3-class research head',
    brain_mri: 'EfficientNetB3',
    lung_ct: 'Lung CNN',
    kidney_us: 'Renal CNN',
    skin_cancer: 'Jeevansh ResNet50',
    pneumonia: 'Jeevansh DenseNet121',
    brain_tumor: 'Jeevansh YOLOv8',
    bone_fracture: 'Jeevansh YOLOv8',
  } as Record<string, string>)[scanType] || 'Diagnostic model';

  const detections = localization?.bounding_boxes || [];

  // Consolidate detections by class so repetitive bounding boxes (e.g. 6 brain_tumor boxes) don't show multiple duplicate lines
  const groupedDetections = (() => {
    const map = new Map<string, { label: string; maxConfidence: number; count: number }>();
    for (const det of detections) {
      const cls = det.class || 'Detected Region';
      const existing = map.get(cls);
      if (!existing) {
        map.set(cls, { label: cls, maxConfidence: det.confidence, count: 1 });
      } else {
        existing.maxConfidence = Math.max(existing.maxConfidence, det.confidence);
        existing.count += 1;
      }
    }
    return Array.from(map.values()).sort((a, b) => b.maxConfidence - a.maxConfidence);
  })();

  return (
    <aside className="result-panel">
      <div className="finding-card">
        <div className="finding-card__top">
          <p className="eyebrow">{isDetection ? 'Detection Model · secondary' : 'Experimental classifier · secondary'}</p>
          {!isDetection && severity && (
            <span className={`severity-badge severity-badge--${severity.toLowerCase()}`}>{severity}</span>
          )}
        </div>
        <h2>{isDetection ? (detections.length === 1 ? '1 Detection' : `${detections.length} Detections`) : top_label}</h2>
        {!isDetection && confidence !== null && (
          <div className="confidence-row">
            <span>Uncalibrated model score</span><strong>{(confidence * 100).toFixed(1)}%</strong>
          </div>
        )}
        {!isDetection && confidence !== null && (
          <div className="confidence-track"><span style={{ width: `${Math.min(confidence * 100, 100)}%` }} /></div>
        )}
        {classification.is_low_confidence && (
          <p className="confidence-warning">Low match strength. Interpret with added caution.</p>
        )}
      </div>

      {isDetection ? (
        <div className="scores-card">
          <div className="card-title-row">
            <p className="eyebrow">Detected objects</p>
            <span>{groupedDetections.length === 1 ? `${detections.length} findings` : `${groupedDetections.length} findings`}</span>
          </div>
          <div className="score-list">
            {groupedDetections.length === 0 ? (
              <div className="history-empty" style={{ padding: '1rem', textAlign: 'center' }}>
                <p>No objects detected.</p>
              </div>
            ) : (
              groupedDetections.map((item, i) => (
                <div key={i} className="is-primary">
                  <span>
                    {item.label}
                    {item.count > 1 ? ` (${item.count} findings)` : ''}
                  </span>
                  <i><b style={{ width: `${Math.min(item.maxConfidence * 100, 100)}%` }} /></i>
                  <strong>{(item.maxConfidence * 100).toFixed(1)}%</strong>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="scores-card">
          <div className="card-title-row"><p className="eyebrow">Experimental class comparison</p><span>{scores.length} classes</span></div>
          <div className="score-list">
            {scores.map(([label, score]) => (
              <div className={label === top_label ? 'is-primary' : ''} key={label}>
                <span>{label}</span>
                <i><b style={{ width: `${Math.min(score * 100, 100)}%` }} /></i>
                <strong>{(score * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      <dl className="model-meta">
        <div><dt>Model</dt><dd>{modelName}</dd></div>
        <div><dt>Processing</dt><dd>{analysisTimeMs ? `${(analysisTimeMs / 1000).toFixed(1)} sec` : 'Archived'}</dd></div>
        <div><dt>Role</dt><dd>Secondary research signal</dd></div>
      </dl>
    </aside>
  );
}
