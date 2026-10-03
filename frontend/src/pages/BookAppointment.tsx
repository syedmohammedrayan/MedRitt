import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { bookAppointment, getDepartments, getDoctors } from '../api/client';
import type { Department, Doctor } from '../types';

const DepartmentIcon = ({ name }: { name: string }) => {
  switch (name?.toLowerCase()) {
    case 'pulmonology':
      return (
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 4c-1.5-1-3-1-4.5 0a4.5 4.5 0 0 0-2 6c0 4 3 6 6.5 10 3.5-4 6.5-6 6.5-10a4.5 4.5 0 0 0-2-6c-1.5-1-3-1-4.5 0Z" opacity="0.3" fill="currentColor"/>
          <path d="M12 2v6" />
          <path d="M12 6c-1.5 1.5-3 2-4.5 2" />
          <path d="M12 6c1.5 1.5 3 2 4.5 2" />
          <path d="M7 11v2" />
          <path d="M17 11v2" />
        </svg>
      );
    case 'neurology':
      return (
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9.5 3A4.5 4.5 0 0 0 5 7.5c0 1.2.5 2.3 1.2 3.1-.7.6-1.2 1.5-1.2 2.4a3.5 3.5 0 0 0 5.4 2.9l.6.4v3.2c0 .8.7 1.5 1.5 1.5h3c.8 0 1.5-.7 1.5-1.5V17" />
          <path d="M14.5 3A4.5 4.5 0 0 1 19 7.5c0 1.2-.5 2.3-1.2 3.1.7.6 1.2 1.5 1.2 2.4a3.5 3.5 0 0 1-5.4 2.9l-.6.4v3.2" />
          <path d="M9.5 3c2 0 3 1.5 3 3v8" />
          <path d="M14.5 3c-2 0-3 1.5-3 3" />
        </svg>
      );
    case 'orthopedics':
      return (
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17 4a3 3 0 0 0-4.2-1.3A3 3 0 0 0 7 4a3 3 0 0 0-.2 4.2L15.6 17a3 3 0 0 0 4.2 1.3A3 3 0 0 0 21 17a3 3 0 0 0 .2-4.2L12.4 4z" />
          <path d="M8 12 12 8" />
        </svg>
      );
    case 'dermatology':
      return (
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
          <path d="M11 8a3 3 0 0 0-3 3" />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M12 8v8" />
          <path d="M8 12h8" />
        </svg>
      );
  }
};

export default function BookAppointment() {
  const navigate = useNavigate();
  const [departments, setDepartments] = useState<Department[]>([]);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [departmentId, setDepartmentId] = useState<number | null>(null);
  const [doctorId, setDoctorId] = useState<number | null>(null);
  const [reason, setReason] = useState('');
  const [scheduledAt, setScheduledAt] = useState('');
  const [busy, setBusy] = useState(false);
  const [loadingDoctors, setLoadingDoctors] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getDepartments()
      .then((data) => {
        setDepartments(data.filter((item) => item.name !== 'Radiology'));
      })
      .catch(() => setError('Could not load departments.'));
  }, []);

  useEffect(() => {
    if (!departmentId) {
      setDoctors([]);
      return;
    }
    setLoadingDoctors(true);
    getDoctors(departmentId)
      .then((data) => {
        setDoctors(data);
      })
      .catch(() => setError('Could not load doctors.'))
      .finally(() => setLoadingDoctors(false));
  }, [departmentId]);
  const selectedDoctor = useMemo(() => doctors.find((item) => item.id === doctorId), [doctors, doctorId]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!departmentId || !doctorId || !scheduledAt) return;
    setBusy(true); setError('');
    try {
      await bookAppointment({ doctor_id: doctorId, department_id: departmentId, reason, scheduled_at: new Date(scheduledAt).toISOString() });
      navigate('/patient/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not book this appointment.');
    } finally { setBusy(false); }
  };

  return (
    <div className="workspace-page portal-page narrow-page">
      <header className="portal-hero"><div><p className="eyebrow">New consultation</p><h1>Choose the right care team.</h1><p>Select a department, specialist, and preferred time.</p></div></header>
      <form className="booking-flow" onSubmit={submit}>
        <section className="portal-card">
          <header><span className="step-number">01</span><div><p className="eyebrow">Department</p><h2>Where should we begin?</h2></div></header>
          <div className="department-grid">
            {departments.filter((item) => item.name !== 'Radiology').map((item) => (
              <button
                type="button"
                key={item.id}
                className={departmentId === item.id ? 'department-card is-selected' : 'department-card'}
                onClick={() => { setDepartmentId(item.id); setDoctorId(null); }}
              >
                <span className="department-icon">
                  <DepartmentIcon name={item.icon} />
                </span>
                <strong>{item.name}</strong>
                <small>{item.description}</small>
              </button>
            ))}
          </div>
        </section>
        <section className={`portal-card${departmentId ? '' : ' is-disabled'}`}>
          <header><span className="step-number">02</span><div><p className="eyebrow">Specialist</p><h2>Select your doctor</h2></div></header>
          <div className="doctor-grid">
            {loadingDoctors && <div className="portal-empty">Loading available doctors…</div>}
            {!loadingDoctors && doctors.map((doctor) => {
              const parts = (doctor.full_name || '').trim().split(' ');
              const initial = parts.length > 0 && parts[parts.length - 1] ? parts[parts.length - 1][0] : 'D';
              return (
                <button
                  type="button"
                  disabled={!doctor.is_available}
                  key={doctor.id}
                  className={`${doctorId === doctor.id ? 'doctor-card is-selected' : 'doctor-card'}${doctor.is_available ? '' : ' is-unavailable'}`}
                  onClick={() => setDoctorId(doctor.id)}
                >
                  <span className="profile-avatar">{initial}</span>
                  <span>
                    <strong>{doctor.full_name}</strong>
                    <small>{doctor.qualification || 'Medical qualification not listed'}</small>
                    <small>{doctor.specialization} · {doctor.department?.name || doctor.department_name}</small>
                    {!doctor.is_available && <em>{doctor.availability_note || 'Temporarily unavailable'}</em>}
                  </span>
                  <i />
                </button>
              );
            })}
            {!loadingDoctors && departmentId && !doctors.length && <div className="portal-empty">No doctors available in this department.</div>}
          </div>
        </section>
        <section className={`portal-card${selectedDoctor ? '' : ' is-disabled'}`}>
          <header><span className="step-number">03</span><div><p className="eyebrow">Visit details</p><h2>Tell us what brings you in</h2></div></header>
          <div className="form-grid">
            <label className="field field--wide"><span>Reason / symptoms</span><textarea value={reason} onChange={(event) => setReason(event.target.value)} rows={4} placeholder="Describe your concern, symptoms, and how long you have had them…" required minLength={3} /></label>
            <label className="field"><span>Preferred date and time</span><input type="datetime-local" value={scheduledAt} min={new Date().toISOString().slice(0, 16)} onChange={(event) => setScheduledAt(event.target.value)} required /></label>
            <div className="booking-summary"><span>Consulting</span><strong>{selectedDoctor?.full_name || 'Select a doctor'}</strong><small>{selectedDoctor ? `${selectedDoctor.qualification} · ${selectedDoctor.specialization}` : 'Degree and specialty will appear here'}</small></div>
          </div>
        </section>
        {error && <div className="form-error">{error}</div>}
        <div className="flow-actions"><button type="button" className="button" onClick={() => navigate(-1)}>Cancel</button><button className="button button--primary" disabled={busy || !selectedDoctor}>{busy ? 'Booking…' : 'Request appointment'}</button></div>
      </form>
    </div>
  );
}
