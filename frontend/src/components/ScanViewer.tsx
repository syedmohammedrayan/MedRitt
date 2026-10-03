import { useState } from 'react';

interface ScanViewerProps {
  scanImageUrl: string;
  heatmapUrl?: string | null;
  overlayUrl?: string | null;
  scanType: string;
  taskType?: string;
  heatmapTargetLabel?: string;
}

type ViewMode = 'original' | 'overlay' | 'compare';

export default function ScanViewer({ scanImageUrl, heatmapUrl, overlayUrl, scanType, taskType = 'classification', heatmapTargetLabel }: ScanViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  
  const scanLabel = ({
    chest_xray: 'Chest X-ray',
    brain_mri: 'Brain MRI',
    lung_ct: 'Lung CT',
    kidney_us: 'Kidney ultrasound',
    skin_cancer: 'Dermatoscopic Image',
    pneumonia: 'Chest X-ray',
    brain_tumor: 'Brain MRI',
    bone_fracture: 'Bone X-ray'
  } as Record<string, string>)[scanType] || 'Diagnostic image';
  
  const isDetection = taskType === 'detection';
  const attributionLabel = isDetection ? 'Detection overlay' : (scanType === 'chest_xray' ? 'RAD-DINO attribution' : 'Model heatmap');
  const activeOverlayUrl = isDetection ? overlayUrl : heatmapUrl;

  const validOverlay = activeOverlayUrl ? activeOverlayUrl : scanImageUrl;

  return (
    <section className="scan-viewer">
      <header className="scan-viewer__header">
        <div>
          <p className="eyebrow eyebrow--light">Image review</p>
          <h2>{scanLabel}</h2>
        </div>
        <div className="viewer-toggle" role="group" aria-label="Image display mode">
          {(['original', 'overlay', 'compare'] as ViewMode[]).map((mode) => (
            <button key={mode} className={viewMode === mode ? 'active' : ''} onClick={() => setViewMode(mode)}>
              {mode === 'overlay' ? attributionLabel : mode[0].toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      </header>

      <div className={`scan-canvas scan-canvas--${viewMode}`}>
        {viewMode === 'compare' ? (
          <>
            <figure><img src={scanImageUrl} alt={`Original ${scanLabel}`} /><figcaption>Original</figcaption></figure>
            <figure><img src={validOverlay} alt={`${scanLabel} ${attributionLabel}`} /><figcaption>{attributionLabel}</figcaption></figure>
          </>
        ) : (
          <figure>
            <img
              src={viewMode === 'original' ? scanImageUrl : validOverlay}
              alt={viewMode === 'original' ? `Original ${scanLabel}` : `${scanLabel} ${attributionLabel}`}
            />
          </figure>
        )}
        <span className="viewer-corner viewer-corner--tl" /><span className="viewer-corner viewer-corner--tr" />
        <span className="viewer-corner viewer-corner--bl" /><span className="viewer-corner viewer-corner--br" />
      </div>

      <footer className="scan-viewer__footer">
        <span><i className="status-pulse" />Study loaded</span>
        {viewMode !== 'original' && !isDetection && (
          <span>{attributionLabel} target · {heatmapTargetLabel || 'primary finding'}</span>
        )}
        {viewMode !== 'original' && isDetection && (
          <span>Bounding box detections</span>
        )}
        <span>For clinician review</span>
      </footer>
    </section>
  );
}
