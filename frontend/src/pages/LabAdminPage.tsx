import React, { useEffect, useState } from 'react';
import { getAdminLabTechs, createLabTech, deleteLabTech } from '../api/client';
import type { UserSummary } from '../types';

export default function LabAdminPage() {
  const emptyLabTech = {
    username: '', password: '', full_name: '', email: '', phone: '',
  };
  const [labTechs, setLabTechs] = useState<UserSummary[]>([]);
  const [form, setForm] = useState(emptyLabTech);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const load = async () => {
    try {
      const items = await getAdminLabTechs();
      setLabTechs(items);
    } catch {
      setError('Could not load lab tech administration.');
    }
  };

  useEffect(() => { load(); }, []);

  const addLabTech = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(''); setMessage('');
    try {
      await createLabTech(form);
      setForm(emptyLabTech);
      setMessage('Lab tech account created successfully.');
      await load();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not create lab tech.');
    } finally {
      setBusy(false);
    }
  };

  const removeLabTech = async (id: number) => {
    if (!window.confirm('Are you sure you want to deactivate this lab tech?')) return;
    setBusy(true); setError(''); setMessage('');
    try {
      await deleteLabTech(id);
      setMessage('Lab tech deactivated safely.');
      await load();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not deactivate lab tech.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="workspace-page portal-page admin-doctors-page">
      <header className="portal-hero">
        <div>
          <p className="eyebrow">Hospital administration</p>
          <h1>Lab Techs admin panel</h1>
          <p>Add lab technicians to manage imaging systems and diagnostics.</p>
        </div>
        <span className="env-badge"><i /> Admin controls</span>
      </header>

      {(error || message) && <div className={error ? 'form-error' : 'success-banner'}>{error || message}</div>}

      <section className="portal-card doctor-create-card">
        <header>
          <div>
            <p className="eyebrow">New Staff</p>
            <h2>Add a lab tech</h2>
          </div>
        </header>
        <form className="form-grid" onSubmit={addLabTech}>
          <label className="field"><span>Full name</span><input required minLength={2} placeholder="Full Name" value={form.full_name} onChange={(event) => setForm({ ...form, full_name: event.target.value })} /></label>
          <label className="field"><span>Login username</span><input required minLength={3} value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} /></label>
          <label className="field"><span>Initial password</span><input required minLength={6} type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} /></label>
          <label className="field"><span>Email</span><input type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} /></label>
          <label className="field"><span>Phone</span><input value={form.phone} onChange={(event) => setForm({ ...form, phone: event.target.value })} /></label>
          <div className="flow-actions field--wide"><button className="button button--primary" disabled={busy}>{busy ? 'Saving...' : 'Add lab tech'}</button></div>
        </form>
      </section>

      <section className="portal-card">
        <header>
          <div>
            <p className="eyebrow">Directory control</p>
            <h2>All lab techs</h2>
          </div>
          <span className="status-pill">{labTechs.filter((item) => item.is_active).length} active</span>
        </header>
        <div className="doctor-admin-list">
          {labTechs.map((tech) => (
            <article key={tech.id} className={!tech.is_active ? 'is-inactive' : ''}>
              <div className="doctor-admin-identity">
                <span className="profile-avatar">{tech.full_name.charAt(0)}</span>
                <div>
                  <strong>{tech.full_name}</strong>
                  <small>@{tech.username} &bull; {tech.email || 'No email'}</small>
                </div>
              </div>
              <span className={`status-pill ${tech.is_active ? 'status-pill--available' : ''}`}>{!tech.is_active ? 'Deactivated' : 'Active'}</span>

              <div className="doctor-admin-actions">
                {tech.is_active && (
                  <button className="text-button text-button--danger" disabled={busy} onClick={() => removeLabTech(tech.id)}>Delete Lab Tech</button>
                )}
              </div>
            </article>
          ))}
          {labTechs.length === 0 && <p className="portal-empty">No lab techs in the system.</p>}
        </div>
      </section>
    </div>
  );
}
