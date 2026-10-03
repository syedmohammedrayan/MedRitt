import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  downloadCaseStudyPdf,
  downloadPdf,
  getMyCaseStudies,
  getMyDiagnosticOrders,
  triggerPdfDownload,
} from '../api/client';
import { useAuth } from '../hooks/useAuth';
import type { CaseStudy, DiagnosticOrder } from '../types';

interface UnifiedReportRecord {
  id: string;
  type: 'diagnostic_scan' | 'case_study';
  title: string;
  category: string;
  doctorName: string;
  doctorSpecialty?: string;
  dateStr: string | null;
  timestamp: number;
  status: string;
  isApproved: boolean;
  clinicalNote: string;
  scanId?: string;
  caseStudyId?: number;
  linkedScansCount?: number;
}

const SCAN_TYPE_NAMES: Record<string, string> = {
  skin_cancer: 'Skin Cancer Classification (Dermatoscopy)',
  pneumonia: 'Pneumonia Detection (Chest Radiography)',
  brain_tumor: 'Brain Tumor Detection (MRI)',
  bone_fracture: 'Bone Fracture Detection (X-Ray)',
};

function formatDate(iso: string | null): string {
  if (!iso) return 'Date not specified';
  try {
    return new Intl.DateTimeFormat('en-IN', {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(iso));
  } catch {
    return String(iso);
  }
}

export default function PatientReportsPage() {
  const { user } = useAuth();
  const [orders, setOrders] = useState<DiagnosticOrder[]>([]);
  const [cases, setCases] = useState<CaseStudy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  // Filters and Sorting
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc'); // Default past to present serially
  const [typeFilter, setTypeFilter] = useState<'all' | 'diagnostic_scan' | 'case_study'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    Promise.all([getMyDiagnosticOrders(), getMyCaseStudies()])
      .then(([nextOrders, nextCases]) => {
        setOrders(nextOrders);
        setCases(nextCases);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || 'Could not load your diagnostic reports.');
      })
      .finally(() => setLoading(false));
  }, [user]);

  // Download diagnostic PDF report
  const downloadReportPdf = async (scanId: string) => {
    setDownloadingId(`scan-${scanId}`);
    setError('');
    try {
      const blob = await downloadPdf(scanId);
      triggerPdfDownload(blob, scanId);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not download the diagnostic report.');
    } finally {
      setDownloadingId(null);
    }
  };

  // Download case study PDF
  const downloadCasePdf = async (caseId: number) => {
    setDownloadingId(`case-${caseId}`);
    setError('');
    try {
      const blob = await downloadCaseStudyPdf(caseId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `MedRittAI_CaseStudy_${caseId}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not download the case study.');
    } finally {
      setDownloadingId(null);
    }
  };

  // Combine and sort chronologically
  const unifiedRecords: UnifiedReportRecord[] = useMemo(() => {
    const list: UnifiedReportRecord[] = [];

    // Diagnostic scan orders
    for (const order of orders) {
      if (!order.scan_id) continue;
      const ts = order.created_at ? new Date(order.created_at).getTime() : 0;
      const scanTypePretty = SCAN_TYPE_NAMES[order.scan_type] || order.scan_type.replace('_', ' ').toUpperCase();
      list.push({
        id: `order-${order.id}`,
        type: 'diagnostic_scan',
        title: scanTypePretty,
        category: 'AI Diagnostic Study',
        doctorName: order.ordering_doctor?.full_name || 'Attending Physician',
        doctorSpecialty: order.ordering_doctor?.specialization || 'Clinical Imaging',
        dateStr: order.created_at,
        timestamp: ts,
        status: order.status,
        isApproved: order.status === 'reviewed',
        clinicalNote: order.clinical_notes || 'Routine imaging study ordered during consultation.',
        scanId: order.scan_id,
      });
    }

    // Integrated Case Studies
    for (const cs of cases) {
      const ts = cs.created_at ? new Date(cs.created_at).getTime() : 0;
      list.push({
        id: `case-${cs.id}`,
        type: 'case_study',
        title: `Comprehensive Case Record #${cs.id}`,
        category: 'Integrated Case Study',
        doctorName: cs.doctor_notes ? 'Attending Clinician' : 'Hospital Medical Board',
        doctorSpecialty: 'Integrated Medicine',
        dateStr: cs.created_at,
        timestamp: ts,
        status: cs.status,
        isApproved: cs.status === 'final',
        clinicalNote: cs.diagnosis || cs.chief_complaint || 'Integrated patient care journey summary.',
        caseStudyId: cs.id,
        linkedScansCount: cs.scan_ids?.length || 0,
      });
    }

    // Sort past to present (asc) or present to past (desc)
    list.sort((a, b) => {
      if (sortOrder === 'asc') return a.timestamp - b.timestamp;
      return b.timestamp - a.timestamp;
    });

    return list;
  }, [orders, cases, sortOrder]);

  // Apply filters and search
  const filteredRecords = useMemo(() => {
    return unifiedRecords.filter((record) => {
      if (typeFilter !== 'all' && record.type !== typeFilter) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesTitle = record.title.toLowerCase().includes(q);
        const matchesDoctor = record.doctorName.toLowerCase().includes(q);
        const matchesNote = record.clinicalNote.toLowerCase().includes(q);
        const matchesStatus = record.status.toLowerCase().includes(q);
        if (!matchesTitle && !matchesDoctor && !matchesNote && !matchesStatus) return false;
      }
      return true;
    });
  }, [unifiedRecords, typeFilter, searchQuery]);

  return (
    <div className="workspace-page portal-page patient-reports-page">
      <header className="portal-hero">
        <div>
          <p className="eyebrow">Medical History & Diagnostics</p>
          <h1>Reports Track</h1>
          <p>Serial chronological track of all your diagnostic studies, imaging scans, and clinical case records from past to present.</p>
        </div>
        <Link className="button button--primary" to="/patient/book-appointment">
          ＋ Request new consultation
        </Link>
      </header>

      {error && <div className="form-error" style={{ marginBottom: '16px' }}>{error}</div>}

      {/* Metric Cards */}
      <section className="metric-grid" style={{ marginBottom: '24px' }}>
        <article>
          <span>Total records</span>
          <strong>{unifiedRecords.length}</strong>
          <small>Tracked in medical history</small>
        </article>
        <article>
          <span>Diagnostic scans</span>
          <strong>{orders.filter((o) => o.scan_id).length}</strong>
          <small>AI-analyzed studies</small>
        </article>
        <article>
          <span>Doctor-approved</span>
          <strong>{unifiedRecords.filter((r) => r.isApproved).length}</strong>
          <small>Clinician verified reports</small>
        </article>
        <article>
          <span>Case studies</span>
          <strong>{cases.length}</strong>
          <small>Integrated patient records</small>
        </article>
      </section>

      {/* Control Bar: Filters & Sorting */}
      <div className="portal-card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', flex: '1 1 360px' }}>
            <input
              type="search"
              placeholder="Search reports by doctor, test type, diagnosis…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                padding: '9px 14px',
                borderRadius: '8px',
                border: '1px solid #cfd6db',
                fontSize: '14px',
                flex: '1 1 240px',
                minWidth: '200px',
              }}
            />
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as any)}
              style={{
                padding: '9px 12px',
                borderRadius: '8px',
                border: '1px solid #cfd6db',
                fontSize: '14px',
                background: '#fff',
                cursor: 'pointer',
              }}
            >
              <option value="all">All record types</option>
              <option value="diagnostic_scan">Diagnostic scans only</option>
              <option value="case_study">Case studies only</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--ink-soft)' }}>Sequence:</span>
            <button
              className={`button ${sortOrder === 'asc' ? 'button--primary' : 'button--outline'}`}
              style={{ padding: '8px 14px', fontSize: '13px', minHeight: '38px' }}
              onClick={() => setSortOrder('asc')}
              title="Track chronologically from earliest past visit to present"
            >
              Past → Present (Serial)
            </button>
            <button
              className={`button ${sortOrder === 'desc' ? 'button--primary' : 'button--outline'}`}
              style={{ padding: '8px 14px', fontSize: '13px', minHeight: '38px' }}
              onClick={() => setSortOrder('desc')}
              title="Show most recent reports first"
            >
              Present → Past (Newest)
            </button>
          </div>
        </div>
      </div>

      {/* Main Serial Timeline List */}
      <section className="portal-card">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <p className="eyebrow">Sequential records</p>
            <h2 style={{ margin: '2px 0' }}>
              Chronological Report Timeline ({filteredRecords.length})
            </h2>
            <p style={{ color: 'var(--ink-soft)', margin: 0, fontSize: '13px' }}>
              {sortOrder === 'asc'
                ? 'Displaying reports serially from your earliest medical visit forward to today.'
                : 'Displaying reports in reverse chronological order (newest first).'}
            </p>
          </div>
          <span className="status-pill">
            {sortOrder === 'asc' ? 'Serial Sequence: #1 to #' + filteredRecords.length : 'Reverse Chronological'}
          </span>
        </header>

        {loading ? (
          <div className="portal-empty" style={{ padding: '40px', textAlign: 'center' }}>
            Loading your medical report history…
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="portal-empty" style={{ padding: '48px', textAlign: 'center' }}>
            <p style={{ fontSize: '16px', fontWeight: 600, marginBottom: '6px' }}>No reports match your filters.</p>
            <p style={{ color: 'var(--ink-soft)', fontSize: '14px', margin: 0 }}>
              Completed diagnostic studies and doctor-approved clinical reports will appear here automatically.
            </p>
          </div>
        ) : (
          <div className="record-list" style={{ gap: '16px' }}>
            {filteredRecords.map((item, index) => {
              // Serial order number calculation
              const serialNumber = sortOrder === 'asc' ? index + 1 : filteredRecords.length - index;

              return (
                <article
                  key={item.id}
                  className="record-row"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '60px minmax(0, 1fr) auto',
                    gap: '16px',
                    alignItems: 'start',
                    padding: '20px',
                    border: '1px solid #cfd6db',
                    borderRadius: '12px',
                    background: '#fff',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.03)',
                  }}
                >
                  {/* Serial Sequence Stamp */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: item.type === 'case_study' ? '#eef2ff' : '#e8f4f8',
                      color: item.type === 'case_study' ? '#3730a3' : '#0a5a68',
                      padding: '10px 4px',
                      borderRadius: '8px',
                      fontWeight: 800,
                      textAlign: 'center',
                    }}
                  >
                    <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Record</span>
                    <span style={{ fontSize: '20px', lineHeight: 1.1 }}>#{String(serialNumber).padStart(2, '0')}</span>
                  </div>

                  {/* Report Details */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                      <span
                        className="status-pill"
                        style={{
                          fontSize: '11px',
                          textTransform: 'uppercase',
                          fontWeight: 700,
                          padding: '3px 8px',
                          background: item.type === 'case_study' ? '#f3e8ff' : '#e0f2fe',
                          color: item.type === 'case_study' ? '#6b21a8' : '#0369a1',
                        }}
                      >
                        {item.category}
                      </span>
                      {item.isApproved && (
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '12px',
                            fontWeight: 700,
                            color: '#047857',
                            background: '#ecfdf5',
                            padding: '3px 8px',
                            borderRadius: '12px',
                          }}
                        >
                          ✓ Doctor Approved
                        </span>
                      )}
                      <span style={{ fontSize: '13px', color: '#64748b' }}>
                        {formatDate(item.dateStr)}
                      </span>
                    </div>

                    <strong style={{ fontSize: '18px', color: '#0f172a', lineHeight: 1.3 }}>
                      {item.title}
                    </strong>

                    <p style={{ margin: 0, fontSize: '14px', color: '#334155' }}>
                      <strong>Clinician:</strong> {item.doctorName}
                      {item.doctorSpecialty ? ` (${item.doctorSpecialty})` : ''}
                    </p>

                    <div
                      style={{
                        marginTop: '6px',
                        padding: '10px 14px',
                        background: '#f8fafc',
                        borderLeft: '3px solid #64748b',
                        borderRadius: '0 6px 6px 0',
                        fontSize: '13px',
                        color: '#1e293b',
                      }}
                    >
                      <span style={{ fontWeight: 700, display: 'block', fontSize: '11px', color: '#64748b', textTransform: 'uppercase', marginBottom: '2px' }}>
                        {item.type === 'case_study' ? 'Clinical Assessment / Diagnosis' : 'Indication & Notes'}
                      </span>
                      {item.clinicalNote}
                      {item.linkedScansCount ? (
                        <span style={{ display: 'block', marginTop: '4px', fontSize: '12px', color: '#4338ca', fontWeight: 600 }}>
                          🔗 {item.linkedScansCount} linked imaging scans attached to this case
                        </span>
                      ) : null}
                    </div>
                  </div>

                  {/* Actions Column */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px',
                      alignItems: 'stretch',
                      minWidth: '150px',
                    }}
                  >
                    {item.type === 'diagnostic_scan' && item.scanId && (
                      <>
                        <Link
                          to={`/results/${item.scanId}`}
                          className="button button--outline"
                          style={{ textAlign: 'center', fontSize: '13px', padding: '9px 12px' }}
                        >
                          View Study Report →
                        </Link>
                        <button
                          className="button button--primary"
                          style={{ fontSize: '13px', padding: '9px 12px' }}
                          disabled={downloadingId === `scan-${item.scanId}`}
                          onClick={() => downloadReportPdf(item.scanId!)}
                        >
                          {downloadingId === `scan-${item.scanId}` ? 'Downloading…' : 'Download PDF ↓'}
                        </button>
                      </>
                    )}

                    {item.type === 'case_study' && item.caseStudyId && (
                      <>
                        <Link
                          to={`/patient/case-study/${item.caseStudyId}`}
                          className="button button--outline"
                          style={{ textAlign: 'center', fontSize: '13px', padding: '9px 12px' }}
                        >
                          View Case Record →
                        </Link>
                        <button
                          className="button button--primary"
                          style={{ fontSize: '13px', padding: '9px 12px' }}
                          disabled={downloadingId === `case-${item.caseStudyId}`}
                          onClick={() => downloadCasePdf(item.caseStudyId!)}
                        >
                          {downloadingId === `case-${item.caseStudyId}` ? 'Downloading…' : 'Download PDF ↓'}
                        </button>
                      </>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
