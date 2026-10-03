import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { downloadCaseStudyPdf, finalizeCaseStudy, getCaseStudy, updateAppointmentStatus } from '../api/client';
import { useAuth } from '../hooks/useAuth';
import type { CaseStudy } from '../types';

export default function CaseStudyView() {
  const { caseStudyId } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [caseStudy, setCaseStudy] = useState<CaseStudy | null>(null);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const id = Number(String(caseStudyId || '').trim());
  useEffect(() => {
    if (!id || isNaN(id)) {
      setError('Invalid case study identifier.');
      return;
    }
    getCaseStudy(id)
      .then(setCaseStudy)
      .catch((err) => setError(err.response?.data?.detail || 'Case study not found or access denied.'));
  }, [id]);

  const finalize = async () => {
    setBusy(true);
    setError('');
    try {
      const updated = await finalizeCaseStudy(id);
      setCaseStudy(updated);
      setSuccessMessage('Case study finalized and appointment marked completed successfully.');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not finalize this case.');
    } finally {
      setBusy(false);
    }
  };

  const completePatientMeet = async () => {
    setBusy(true);
    setError('');
    try {
      if (caseStudy?.status !== 'final') {
        await finalizeCaseStudy(id);
      }
      if (caseStudy?.appointment_id) {
        await updateAppointmentStatus(caseStudy.appointment_id, 'completed');
      }
      navigate('/doctor/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not complete the patient meet.');
      setBusy(false);
    }
  };

  const download = async () => {
    try {
      const blob = await downloadCaseStudyPdf(id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `MedRittAI_Case_${id}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Could not download the case study.');
    }
  };

  if (!caseStudy) {
    return (
      <div className="workspace-page portal-page">
        <header className="portal-hero">
          <div>
            <p className="eyebrow">Clinical case study</p>
            <h1>Case study <em>#{caseStudyId}</em></h1>
          </div>
          <Link to={user?.role === 'doctor' ? '/doctor/dashboard' : '/patient/dashboard'} className="button button--outline">
            ← Return to dashboard
          </Link>
        </header>
        {error ? (
          <div className="portal-card" style={{ marginTop: '20px' }}>
            <div className="form-error" style={{ marginBottom: '16px' }}>{error}</div>
            <p style={{ color: 'var(--ink-soft)' }}>
              This case study may still be in draft, assigned to another clinician, or does not exist.
            </p>
            <div style={{ marginTop: '16px' }}>
              <Link to={user?.role === 'doctor' ? '/doctor/dashboard' : '/patient/dashboard'} className="button button--primary">
                Return to queue
              </Link>
            </div>
          </div>
        ) : (
          <div className="portal-card" style={{ marginTop: '20px', padding: '32px', textAlign: 'center' }}>
            <div className="status-pill status-pill--active" style={{ marginBottom: '12px' }}>Loading case study…</div>
            <p style={{ color: 'var(--ink-soft)', margin: 0 }}>Retrieving integrated clinical findings and patient prescriptions.</p>
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="workspace-page portal-page case-study-page">
      <header className="case-study-header">
        <div>
          <p className="eyebrow">Complete patient journey</p>
          <h1>Case study <em>#{caseStudy.id}</em></h1>
          <p>{caseStudy.patient.full_name} · Appointment #{caseStudy.appointment_id}</p>
        </div>
        <div className="flow-actions">
          {user?.role === 'doctor' && (
            <Link to={`/doctor/consultation/${caseStudy.appointment_id}`} className="button button--outline">
              ← Edit Consultation
            </Link>
          )}
          <Link to={user?.role === 'doctor' ? '/doctor/dashboard' : '/patient/dashboard'} className="button button--outline">
            {user?.role === 'doctor' ? 'Clinical Queue' : 'Dashboard'}
          </Link>
          <span className={`status-pill status-${caseStudy.status}`}>{caseStudy.status}</span>
          <button className="button" onClick={download}>Download PDF</button>
          {user?.role === 'doctor' && caseStudy.status !== 'final' && (
            <button className="button button--primary" disabled={busy} onClick={finalize}>
              {busy ? 'Finalizing…' : 'Sign & Finalize'}
            </button>
          )}
        </div>
      </header>
      {error && <div className="form-error" style={{ marginBottom: '16px' }}>{error}</div>}
      {successMessage && <div className="success-banner" style={{ marginBottom: '16px' }}>{successMessage}</div>}
      <div className="case-timeline"><span className="is-complete">Consultation</span><i /><span className={caseStudy.scan_ids.length ? 'is-complete' : ''}>AI diagnostics</span><i /><span className={caseStudy.prescriptions.length ? 'is-complete' : ''}>Treatment</span><i /><span className={caseStudy.status === 'final' ? 'is-complete' : ''}>Final record</span></div>
      <article className="case-paper">
        <header><div><span>MedRittAI Hospital Intelligence</span><strong>Integrated clinical case record</strong></div><dl><div><dt>Patient</dt><dd>{caseStudy.patient.full_name}</dd></div><div><dt>Record</dt><dd>MED-{String(caseStudy.patient.id).padStart(4, '0')}</dd></div><div><dt>Status</dt><dd>{caseStudy.status.toUpperCase()}</dd></div></dl></header>
        {[
          ['Chief complaint', caseStudy.chief_complaint],
          ['Clinical history', caseStudy.clinical_history],
          ['Diagnostic findings', caseStudy.diagnostic_findings],
          ['Clinical diagnosis', caseStudy.diagnosis],
          ['Treatment plan', caseStudy.treatment_plan],
        ].map(([label, value], index) => <section className="case-section" key={label}><span>{String(index + 1).padStart(2, '0')}</span><div><h2>{label}</h2><p>{value || 'Not documented.'}</p></div></section>)}
        <section className="case-section"><span>06</span><div><h2>Diagnostic studies</h2><div className="scan-link-grid">{caseStudy.scan_ids.map((scanId) => <Link key={scanId} to={`/results/${scanId}`}><strong>AI imaging study</strong><small>{scanId.slice(0, 8).toUpperCase()} · Open report →</small></Link>)}{!caseStudy.scan_ids.length && <p>No linked scans.</p>}</div></div></section>
        <section className="case-section"><span>07</span><div><h2>Prescription</h2>{caseStudy.prescriptions.map((prescription) => <div className="prescription-block" key={prescription.id}><p><strong>{prescription.diagnosis}</strong></p>{prescription.medications.map((medication, index) => <div className="medication-line" key={index}><strong>{medication.name}</strong><span>{medication.dosage}</span><span>{medication.frequency}</span><span>{medication.duration}</span></div>)}<small>{prescription.instructions}</small></div>)}{!caseStudy.prescriptions.length && <p>No prescription recorded.</p>}</div></section>
        <section className="case-section"><span>08</span><div><h2>Follow-up plan</h2><p>{caseStudy.follow_up_plan || 'Not documented.'}</p><div className="doctor-signature"><span>Clinician notes</span><strong>{caseStudy.doctor_notes || 'No additional notes.'}</strong></div></div></section>
      </article>

      {user?.role === 'doctor' && (
        <section className="portal-card" style={{ marginTop: '24px' }}>
          <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <p className="eyebrow">Consultation Completion</p>
              <h2 style={{ margin: '4px 0' }}>{caseStudy.status === 'final' ? 'Patient Consultation Completed' : 'Finish & Complete Patient Meet'}</h2>
              <p style={{ color: 'var(--ink-soft)', margin: 0, fontSize: '13px' }}>
                {caseStudy.status === 'final'
                  ? 'This case study has been signed and finalized. The patient appointment is marked completed.'
                  : 'Review the case details above. Once verified, finalize the case to close this consultation and return to your queue.'}
              </p>
            </div>
            <div className="flow-actions">
              <Link to={`/doctor/consultation/${caseStudy.appointment_id}`} className="button button--outline">
                ← Edit Consultation / Prescriptions
              </Link>
              {caseStudy.status !== 'final' ? (
                <button className="button button--primary" disabled={busy} onClick={completePatientMeet}>
                  {busy ? 'Completing meet…' : 'Sign, Complete Meet & Return to Queue →'}
                </button>
              ) : (
                <Link to="/doctor/dashboard" className="button button--primary">
                  Return to Clinical Queue →
                </Link>
              )}
            </div>
          </header>
        </section>
      )}
    </div>
  );
}
