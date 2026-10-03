import { useState } from 'react';
import { downloadPdf, triggerPdfDownload } from '../api/client';
import type { ReportData } from '../types';
import PatientReport from './PatientReport';

interface PatientFinalReportProps {
  scanId: string;
  report: ReportData;
  heatmapUrl?: string | null;
  overlayUrl?: string | null;
  originalImageUrl: string;
}

const detailSections: Array<{ key: keyof ReportData; title: string }> = [
  { key: 'clinical_history', title: 'Clinical information' },
  { key: 'technique', title: 'Technique' },
  { key: 'image_quality', title: 'Study quality / limitations' },
  { key: 'differential_diagnosis', title: 'Differential considerations' },
  { key: 'recommendations', title: 'Recommendations' },
  { key: 'critical_communication', title: 'Communication' },
];

type ReportImageViewMode = 'both' | 'heatmap' | 'original';

export default function PatientFinalReport({ scanId, report, heatmapUrl, overlayUrl, originalImageUrl }: PatientFinalReportProps) {
  const [downloading, setDownloading] = useState(false);
  const [imageMode, setImageMode] = useState<ReportImageViewMode>('both');
  const [error, setError] = useState('');

  const validOverlay = overlayUrl || heatmapUrl || originalImageUrl;
  const isDetection = !!overlayUrl;
  const attributionLabel = isDetection ? 'Detection overlay' : 'Grad-CAM heatmap';

  const download = async () => {
    setDownloading(true);
    setError('');
    try {
      const blob = await downloadPdf(scanId);
      triggerPdfDownload(blob, scanId);
    } catch {
      setError('The complete final report could not be downloaded. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="patient-final-report">
      <section className="patient-final-summary">
        <header>
          <div><p className="eyebrow">Doctor-approved clinical record</p><h2>Your final imaging report</h2><p>View the reviewed heatmap and clinical conclusions, or save the complete report with every section.</p></div>
          <button className="button button--primary patient-final-download" onClick={() => void download()} disabled={downloading}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0 4-4m-4 4-4-4M5 20h14" /></svg>
            {downloading ? 'Preparing full report…' : 'Download complete final report'}
          </button>
        </header>
        {error && <div className="form-error" role="alert">{error}</div>}

        <div className="patient-final-grid">
          <article className="patient-heatmap-card">
            <div className="patient-heatmap-card__heading">
              <div>
                <span>Reviewed image</span>
                <strong>
                  {imageMode === 'both'
                    ? `${attributionLabel} (Left) & Original scan (Right)`
                    : imageMode === 'heatmap'
                    ? attributionLabel
                    : 'Original scan'}
                </strong>
              </div>
              <div className="patient-image-toggle" role="group" aria-label="Report image view">
                <button className={imageMode === 'both' ? 'active' : ''} onClick={() => setImageMode('both')}>Both scans</button>
                <button className={imageMode === 'heatmap' ? 'active' : ''} onClick={() => setImageMode('heatmap')}>{isDetection ? 'Overlay' : 'Grad-CAM'}</button>
                <button className={imageMode === 'original' ? 'active' : ''} onClick={() => setImageMode('original')}>Original</button>
              </div>
            </div>

            {imageMode === 'both' ? (
              <div className="patient-dual-scans">
                {/* Left side: Grad-CAM */}
                <figure className="patient-scan-panel">
                  <div className="patient-scan-panel__heading">
                    <span>AI Attribution (Left)</span>
                    <strong>{attributionLabel}</strong>
                  </div>
                  <img src={validOverlay} alt={attributionLabel} />
                  <figcaption>
                    {isDetection
                      ? 'Identified pathology / fracture bounding overlay.'
                      : `Highlighted model attention for ${report.heatmap_target_label || report.top_label || 'the primary finding'}.`}
                  </figcaption>
                </figure>

                {/* Right side: Original scan */}
                <figure className="patient-scan-panel">
                  <div className="patient-scan-panel__heading">
                    <span>Reference (Right)</span>
                    <strong>Original diagnostic scan</strong>
                  </div>
                  <img src={originalImageUrl} alt="Original diagnostic scan" />
                  <figcaption>Original unprocessed diagnostic scan supplied for this report.</figcaption>
                </figure>
              </div>
            ) : (
              <figure>
                <img
                  src={imageMode === 'heatmap' ? validOverlay : originalImageUrl}
                  alt={imageMode === 'heatmap' ? attributionLabel : 'Original diagnostic scan'}
                />
                <figcaption>
                  {imageMode === 'heatmap'
                    ? (isDetection
                      ? 'Bounding box detections.'
                      : `Highlighted model attention for ${report.heatmap_target_label || report.top_label || 'the primary finding'}.`)
                    : 'Original diagnostic image supplied for this report.'}
                </figcaption>
              </figure>
            )}
          </article>

          <div className="patient-conclusion-stack">
            <article className="patient-conclusion-card"><span>Findings</span><h3>What was observed</h3><p>{report.findings || 'No findings were documented.'}</p></article>
            <article className="patient-conclusion-card patient-conclusion-card--impression"><span>Impression</span><h3>Final clinical conclusion</h3><p>{report.impression || 'No impression was documented.'}</p></article>
            {report.doctor_assessment && <article className="patient-conclusion-card patient-conclusion-card--doctor"><span>Doctor assessment</span><h3>Clinical sign-off</h3><p>{report.doctor_assessment}</p></article>}
          </div>
        </div>
      </section>

      <section className="patient-full-record portal-card">
        <header><div><p className="eyebrow">Complete record</p><h2>All finalized report details</h2></div><span className="status-pill status-pill--available">Final</span></header>
        <div className="patient-full-record__grid">
          {detailSections.map((section) => {
            const value = report[section.key];
            return <article key={section.key}><span>{section.title}</span><p>{typeof value === 'string' && value.trim() ? value : 'Not documented.'}</p></article>;
          })}
        </div>
        <footer><p>{report.disclaimer}</p><button className="button" onClick={() => void download()} disabled={downloading}>{downloading ? 'Preparing…' : 'Download report with heatmap'}</button></footer>
      </section>

      <PatientReport scanId={scanId} />
    </div>
  );
}
