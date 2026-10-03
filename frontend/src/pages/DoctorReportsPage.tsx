import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { getDoctorReports } from '../api/client';
import { useAuth } from '../hooks/useAuth';
import type { DoctorReportQuery, DoctorReportStatus, DoctorReportSummary } from '../types';

const SCAN_TYPE_LABELS: Record<string, string> = {
  skin_cancer: 'Skin lesion analysis',
  pneumonia: 'Pneumonia (chest X-ray)',
  brain_tumor: 'Brain tumour (MRI)',
  bone_fracture: 'Bone fracture (X-ray)',
};

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function scanTypeLabel(scanType: string): string {
  return SCAN_TYPE_LABELS[scanType] || scanType.replace(/_/g, ' ');
}

/** Format the stored YYYY-MM-DD test date without timezone conversion. */
function formatDate(value: string | null): string {
  if (!value) return 'Date not recorded';
  const [year, month, day] = value.split('-').map(Number);
  if (!year || !month || !day) return value;
  return `${day} ${MONTHS[month - 1]} ${year}`;
}

function findingText(report: DoctorReportSummary): string {
  if (report.task_type === 'detection') {
    const detection = report.detection;
    if (!detection || detection.count === 0) return 'No detections';
    const classes = detection.classes.length ? ` · ${detection.classes.join(', ')}` : '';
    return `${detection.count} detection${detection.count === 1 ? '' : 's'}${classes}`;
  }
  if (!report.top_label) return 'Finding not available';
  const confidence = report.confidence != null ? ` · ${(report.confidence * 100).toFixed(1)}%` : '';
  return `${report.top_label}${confidence}`;
}

interface PatientGroup {
  patientId: number;
  fullName: string;
  username: string;
  reports: DoctorReportSummary[];
  firstDate: string | null;
  latestDate: string | null;
  latestKey: string;
}

function sortKey(report: DoctorReportSummary): string {
  return `${report.tested_at || ''}#${String(report.report_id).padStart(10, '0')}`;
}

function groupByPatient(reports: DoctorReportSummary[], sort: 'newest' | 'oldest'): PatientGroup[] {
  const groups = new Map<number, PatientGroup>();
  for (const report of reports) {
    const existing = groups.get(report.patient.id);
    if (existing) existing.reports.push(report);
    else groups.set(report.patient.id, {
      patientId: report.patient.id,
      fullName: report.patient.full_name,
      username: report.patient.username,
      reports: [report],
      firstDate: null,
      latestDate: null,
      latestKey: '',
    });
  }
  const result = Array.from(groups.values()).map((group) => {
    const chronological = [...group.reports].sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
    const first = chronological[0];
    const latest = chronological[chronological.length - 1];
    return {
      ...group,
      reports: sort === 'newest' ? chronological.reverse() : chronological,
      firstDate: first?.test_date ?? null,
      latestDate: latest?.test_date ?? null,
      latestKey: latest ? sortKey(latest) : '',
    };
  });
  // Most recently active patient first.
  return result.sort((a, b) => b.latestKey.localeCompare(a.latestKey));
}

export default function DoctorReportsPage() {
  const { user } = useAuth();
  const [searchInput, setSearchInput] = useState('');
  const [search, setSearch] = useState('');
  const [scanType, setScanType] = useState('');
  const [status, setStatus] = useState<'' | DoctorReportStatus>('');
  const [sort, setSort] = useState<'newest' | 'oldest'>('newest');
  const [reports, setReports] = useState<DoctorReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(searchInput.trim()), 300);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    let cancelled = false;
    const query: DoctorReportQuery = { sort };
    if (search) query.search = search;
    if (scanType) query.scan_type = scanType;
    if (status) query.status = status;
    setLoading(true);
    setError('');
    getDoctorReports(query)
      .then((response) => { if (!cancelled) setReports(response.reports); })
      .catch((err) => { if (!cancelled) setError(err.response?.data?.detail || 'Could not load reports.'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [search, scanType, status, sort]);

  const groups = useMemo(() => groupByPatient(reports, sort), [reports, sort]);
  const hasFilters = Boolean(search || scanType || status);

  return (
    <div className="workspace-page portal-page doctor-reports-page">
      <header className="portal-hero">
        <div>
          <p className="eyebrow">Reports</p>
          <h1>Patient reports</h1>
          <p>Review diagnostic reports across your patient panel — every study from the first report to the most recent, grouped by patient.</p>
        </div>
      </header>

      <section className="portal-card doctor-reports-toolbar" aria-label="Report search and filters">
        <label className="field doctor-reports-search">
          <span>Patient</span>
          <input
            id="doctor-reports-search"
            type="search"
            placeholder="Search patient name..."
            value={searchInput}
            onChange={(event) => setSearchInput(event.target.value)}
            autoComplete="off"
          />
        </label>
        <label className="field">
          <span>Test type</span>
          <select id="doctor-reports-scan-type" value={scanType} onChange={(event) => setScanType(event.target.value)}>
            <option value="">All tests</option>
            {Object.entries(SCAN_TYPE_LABELS)
              .filter(([value]) => {
                if (!user?.department_name) return true;
                const deptMap: Record<string, string> = {
                  'Dermatology': 'skin_cancer',
                  'Neurology': 'brain_tumor',
                  'Pulmonology': 'pneumonia',
                  'Orthopedics': 'bone_fracture',
                };
                const allowedType = deptMap[user.department_name];
                return !allowedType || value === allowedType;
              })
              .map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Review status</span>
          <select id="doctor-reports-status" value={status} onChange={(event) => setStatus(event.target.value as '' | DoctorReportStatus)}>
            <option value="">All statuses</option>
            <option value="pending_review">Pending review</option>
            <option value="reviewed">Reviewed</option>
          </select>
        </label>
        <label className="field">
          <span>Order</span>
          <select id="doctor-reports-sort" value={sort} onChange={(event) => setSort(event.target.value as 'newest' | 'oldest')}>
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
      </section>

      {error && <div className="form-error">{error}</div>}

      {loading && !reports.length && <div className="portal-card"><div className="portal-empty">Loading reports…</div></div>}

      {!loading && !error && !groups.length && (
        <div className="portal-card">
          <div className="portal-empty">{hasFilters ? 'No reports found for this patient.' : 'No reports yet.'}</div>
        </div>
      )}

      <div className="doctor-reports-groups">
        {groups.map((group) => (
          <section key={group.patientId} className="portal-card doctor-report-group" aria-label={`Reports for ${group.fullName}`}>
            <header>
              <div>
                <p className="eyebrow">Patient</p>
                <h2>{group.fullName}</h2>
                <small className="doctor-report-group__meta">
                  @{group.username} · ID {group.patientId} · {group.reports.length} report{group.reports.length === 1 ? '' : 's'}
                  {group.firstDate && ` · ${formatDate(group.firstDate)} → ${formatDate(group.latestDate)}`}
                </small>
              </div>
            </header>
            <ol className="doctor-report-timeline">
              {group.reports.map((report) => (
                <li key={report.report_id} className="record-row doctor-report-row">
                  <span className="record-icon">{report.task_type === 'detection' ? 'DX' : 'AI'}</span>
                  <div>
                    <strong>{scanTypeLabel(report.scan_type)} · {report.modality || 'Imaging'}</strong>
                    <small>
                      <time dateTime={report.tested_at || undefined}>{formatDate(report.test_date)} at {report.test_time || '—'}</time>
                      {' · '}{group.fullName}
                    </small>
                    <p>{findingText(report)}</p>
                    <div className="doctor-report-row__pills">
                      <span className={`status-pill${report.doctor_approved ? ' status-pill--available' : ''}`}>
                        {report.doctor_approved ? 'Reviewed' : 'Pending review'}
                      </span>
                      <span className="status-pill">Scan {report.scan_status}</span>
                      {report.forwarded_to_me && <span className="status-pill">Forwarded to you</span>}
                    </div>
                  </div>
                  <Link className="button doctor-report-row__action" to={`/results/${report.scan_id}`}>View report →</Link>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  );
}
