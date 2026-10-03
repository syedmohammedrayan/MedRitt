import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getAdminPharmacyStaff, createPharmacyStaff, deletePharmacyStaff } from '../api/client';
import type { UserSummary } from '../types';

export default function PharmacyAdminPage() {
  const emptyStaff = {
    username: '', password: '', full_name: '', email: '', phone: '',
  };
  const [staffList, setStaffList] = useState<UserSummary[]>([]);
  const [form, setForm] = useState(emptyStaff);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const load = async () => {
    try {
      const items = await getAdminPharmacyStaff();
      setStaffList(items);
    } catch {
      setError('Could not load pharmacy administration.');
    }
  };

  useEffect(() => { load(); }, []);

  const addStaff = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(''); setMessage('');
    try {
      await createPharmacyStaff(form);
      setForm(emptyStaff);
      setMessage('Chemist / Pharmacist account created successfully.');
      await load();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not create pharmacy account.');
    } finally {
      setBusy(false);
    }
  };

  const removeStaff = async (id: number, name: string) => {
    if (!window.confirm(`Are you sure you want to deactivate ${name}?`)) return;
    setBusy(true); setError(''); setMessage('');
    try {
      await deletePharmacyStaff(id);
      setMessage('Pharmacy staff deactivated safely.');
      await load();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Could not deactivate pharmacy account.');
    } finally {
      setBusy(false);
    }
  };

  const activeCount = staffList.filter((item) => item.is_active).length;

  return (
    <div className="workspace-page portal-page admin-doctors-page">
      <header className="portal-hero">
        <div>
          <p className="eyebrow">Hospital administration</p>
          <h1>Chemist &amp; Pharmacy Admin Panel</h1>
          <p>Add pharmacists, chemists, and store managers, maintain pharmacy credentials, and oversee staff access.</p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <Link to="/pharmacy/inventory" className="button button--outline" style={{ fontSize: '12px' }}>
            Store inventory →
          </Link>
          <span className="env-badge"><i /> Admin controls</span>
        </div>
      </header>

      {(error || message) && <div className={error ? 'form-error' : 'success-banner'}>{error || message}</div>}

      <section className="portal-card doctor-create-card">
        <header>
          <div>
            <p className="eyebrow">New Staff</p>
            <h2>Add a chemist / pharmacist</h2>
          </div>
          <span className="status-pill">Hospital pharmacy access</span>
        </header>
        <form className="form-grid" onSubmit={addStaff}>
          <label className="field">
            <span>Full name</span>
            <input
              required
              minLength={2}
              placeholder="e.g. Ramesh Kumar (Chief Pharmacist)"
              value={form.full_name}
              onChange={(event) => setForm({ ...form, full_name: event.target.value })}
            />
          </label>
          <label className="field">
            <span>Login username</span>
            <input
              required
              minLength={3}
              placeholder="e.g. chemist_ramesh"
              value={form.username}
              onChange={(event) => setForm({ ...form, username: event.target.value })}
            />
          </label>
          <label className="field">
            <span>Initial password</span>
            <div className="password-input-wrapper" style={{ position: 'relative' }}>
              <input
                required
                minLength={6}
                type={showPassword ? 'text' : 'password'}
                placeholder="Temporary password"
                value={form.password}
                onChange={(event) => setForm({ ...form, password: event.target.value })}
                style={{ width: '100%', paddingRight: '40px' }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '15px'
                }}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
          </label>
          <label className="field">
            <span>Email</span>
            <input
              type="email"
              placeholder="chemist@medritt.ai"
              value={form.email}
              onChange={(event) => setForm({ ...form, email: event.target.value })}
            />
          </label>
          <label className="field">
            <span>Phone</span>
            <input
              value={form.phone}
              placeholder="+91-XXXXXXXXXX"
              onChange={(event) => setForm({ ...form, phone: event.target.value })}
            />
          </label>
          <div className="flow-actions field--wide">
            <button className="button button--primary" disabled={busy}>
              {busy ? 'Saving...' : 'Add chemist account'}
            </button>
          </div>
        </form>
      </section>

      <section className="portal-card">
        <header>
          <div>
            <p className="eyebrow">Directory control</p>
            <h2>All pharmacy personnel</h2>
          </div>
          <span className="status-pill">{activeCount} active</span>
        </header>
        <div className="doctor-admin-list">
          {staffList.map((staff) => (
            <article key={staff.id} className={!staff.is_active ? 'is-inactive' : ''}>
              <div className="doctor-admin-identity">
                <span className="profile-avatar" style={{ backgroundColor: 'var(--teal-soft, #e6f6f4)', color: 'var(--teal, #0b7c74)' }}>
                  Rx
                </span>
                <div>
                  <strong>{staff.full_name}</strong>
                  <small>@{staff.username} &bull; {staff.email || 'No email'} &bull; {staff.phone || 'No phone'}</small>
                </div>
              </div>
              <div className="doctor-admin-status">
                <span className={`status-pill ${staff.is_active ? 'status-pill--active' : 'status-pill--inactive'}`}>
                  {staff.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="doctor-admin-actions">
                {staff.is_active && (
                  <button
                    type="button"
                    className="button button--outline button--small"
                    style={{ color: 'var(--heat-red, #c0392b)', borderColor: 'rgba(192, 57, 43, 0.3)' }}
                    onClick={() => removeStaff(staff.id, staff.full_name)}
                    disabled={busy}
                  >
                    Deactivate
                  </button>
                )}
              </div>
            </article>
          ))}
          {!staffList.length && (
            <div className="portal-empty" style={{ padding: '24px', textAlign: 'center', color: 'var(--ink-soft)' }}>
              No pharmacy staff registered yet. Use the form above to add the hospital's first chemist.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
