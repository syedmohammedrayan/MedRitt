import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getMyAppointments,
  getMyCaseStudies,
  getMyPharmacyBills,
  getMyPrescriptions,
} from '../api/client';
import { useAuth } from '../hooks/useAuth';
import type { Appointment, CaseStudy, PharmacyBill, Prescription } from '../types';

const currency = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  minimumFractionDigits: 2,
});

function formatDate(iso: string | null): string {
  if (!iso) return 'Date not recorded';
  try {
    return new Intl.DateTimeFormat('en-IN', {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(iso));
  } catch {
    return String(iso);
  }
}

interface VisitPrescriptionGroup {
  visitKey: string;
  serialNumber: number;
  dateStr: string | null;
  timestamp: number;
  appointment?: Appointment;
  doctorName: string;
  doctorSpecialty?: string;
  departmentName?: string;
  reasonForVisit: string;
  diagnosis: string;
  followUpPlan: string;
  prescriptions: Prescription[];
  linkedBills: PharmacyBill[];
  status: 'dispensed' | 'billed' | 'pending';
}

export default function PatientPrescriptionsPage() {
  const { user } = useAuth();
  const [prescriptions, setPrescriptions] = useState<Prescription[]>([]);
  const [bills, setBills] = useState<PharmacyBill[]>([]);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [cases, setCases] = useState<CaseStudy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Filters & Sorting
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc'); // Default past to present
  const [statusFilter, setStatusFilter] = useState<'all' | 'dispensed' | 'pending'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    Promise.all([
      getMyPrescriptions(),
      getMyPharmacyBills(),
      getMyAppointments(),
      getMyCaseStudies(),
    ])
      .then(([nextPrescriptions, nextBills, nextAppointments, nextCases]) => {
        setPrescriptions(nextPrescriptions);
        setBills(nextBills);
        setAppointments(nextAppointments);
        setCases(nextCases);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || 'Could not load your medication history.');
      })
      .finally(() => setLoading(false));
  }, [user]);

  // Map and group prescriptions by Visit / Appointment
  const visitGroups: VisitPrescriptionGroup[] = useMemo(() => {
    const apptMap = new Map<number, Appointment>();
    for (const appt of appointments) {
      apptMap.set(appt.id, appt);
    }

    const caseMap = new Map<number, CaseStudy>();
    for (const cs of cases) {
      if (cs.appointment_id != null) {
        caseMap.set(cs.appointment_id, cs);
      }
    }

    // Map bills to prescription IDs
    const billsByPrescription = new Map<number, PharmacyBill[]>();
    for (const bill of bills) {
      const rxId = bill.prescription?.id;
      if (rxId) {
        const list = billsByPrescription.get(rxId) || [];
        list.push(bill);
        billsByPrescription.set(rxId, list);
      }
    }

    // Group prescriptions by appointment_id (or fallback prescription id)
    const groupsByAppt = new Map<number, Prescription[]>();
    for (const rx of prescriptions) {
      const apptId = rx.appointment_id || 0;
      const list = groupsByAppt.get(apptId) || [];
      list.push(rx);
      groupsByAppt.set(apptId, list);
    }

    const result: VisitPrescriptionGroup[] = [];

    // Build each visit entry
    groupsByAppt.forEach((rxList, apptId) => {
      const appt = apptMap.get(apptId);
      const linkedCase = caseMap.get(apptId);
      const firstRx = rxList[0];

      const dateStr = appt?.scheduled_at || firstRx.created_at;
      const timestamp = dateStr ? new Date(dateStr).getTime() : 0;

      // Collect all bills linked to these prescriptions
      const visitBills: PharmacyBill[] = [];
      for (const rx of rxList) {
        const bList = billsByPrescription.get(rx.id) || [];
        for (const b of bList) {
          if (!visitBills.some((existing) => existing.id === b.id)) {
            visitBills.push(b);
          }
        }
      }

      // Determine overall dispensing status
      let overallStatus: 'dispensed' | 'billed' | 'pending' = 'pending';
      if (visitBills.some((b) => b.status === 'dispensed')) {
        overallStatus = 'dispensed';
      } else if (visitBills.length > 0) {
        overallStatus = 'billed';
      }

      // Extract follow-up plan
      let followUp = linkedCase?.follow_up_plan || '';
      if (!followUp && appt?.notes) {
        const match = appt.notes.match(/follow[- ]?up[:\s]+([^\n\r.]+)/i);
        if (match) followUp = match[1];
      }

      // Extract diagnosis
      const diagnosis =
        rxList.map((r) => r.diagnosis).filter(Boolean).join('; ') ||
        linkedCase?.diagnosis ||
        appt?.reason ||
        'Clinical Consultation';

      result.push({
        visitKey: `visit-${apptId || firstRx.id}`,
        serialNumber: 0, // assigned after sorting
        dateStr,
        timestamp,
        appointment: appt,
        doctorName: appt?.doctor?.full_name || firstRx.doctor?.full_name || 'Attending Physician',
        doctorSpecialty: appt?.doctor?.specialization || firstRx.doctor?.specialization || 'General Medicine',
        departmentName: appt?.department?.name || 'Outpatient Clinic',
        reasonForVisit: appt?.reason || 'Clinical Consultation',
        diagnosis,
        followUpPlan: followUp,
        prescriptions: rxList,
        linkedBills: visitBills,
        status: overallStatus,
      });
    });

    // Sort chronologically (Past to Present default or Present to Past)
    result.sort((a, b) => {
      if (sortOrder === 'asc') return a.timestamp - b.timestamp;
      return b.timestamp - a.timestamp;
    });

    // Assign sequential serial numbers
    result.forEach((item, index) => {
      item.serialNumber = sortOrder === 'asc' ? index + 1 : result.length - index;
    });

    return result;
  }, [prescriptions, bills, appointments, cases, sortOrder]);

  // Filtered visits
  const filteredVisits = useMemo(() => {
    return visitGroups.filter((group) => {
      if (statusFilter === 'dispensed' && group.status !== 'dispensed') return false;
      if (statusFilter === 'pending' && group.status === 'dispensed') return false;

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesDoctor = group.doctorName.toLowerCase().includes(q);
        const matchesDiagnosis = group.diagnosis.toLowerCase().includes(q);
        const matchesMed = group.prescriptions.some((p) =>
          p.medications.some((m) => m.name.toLowerCase().includes(q))
        );
        const matchesFollowUp = group.followUpPlan.toLowerCase().includes(q);
        if (!matchesDoctor && !matchesDiagnosis && !matchesMed && !matchesFollowUp) {
          return false;
        }
      }
      return true;
    });
  }, [visitGroups, statusFilter, searchQuery]);

  // Overall metric counts
  const totalMedicationsCount = useMemo(() => {
    return prescriptions.reduce((sum, p) => sum + p.medications.length, 0);
  }, [prescriptions]);

  const dispensedBillsCount = useMemo(() => {
    return bills.filter((b) => b.status === 'dispensed').length;
  }, [bills]);

  return (
    <div className="workspace-page portal-page patient-prescriptions-page">
      <header className="portal-hero">
        <div>
          <p className="eyebrow">Prescriptions & Pharmacy Store</p>
          <h1>Medicines & Purchases</h1>
          <p>
            Track all doctor-issued prescriptions, medicine dispensary bills, and clinical follow-up plans visit-by-visit in serial order.
          </p>
        </div>
        <Link className="button button--primary" to="/patient/book-appointment">
          ＋ Book doctor consultation
        </Link>
      </header>

      {error && <div className="form-error" style={{ marginBottom: '16px' }}>{error}</div>}

      {/* Metrics Row */}
      <section className="metric-grid" style={{ marginBottom: '24px' }}>
        <article>
          <span>Consultation visits</span>
          <strong>{visitGroups.length}</strong>
          <small>Visits with prescriptions</small>
        </article>
        <article>
          <span>Medications prescribed</span>
          <strong>{totalMedicationsCount}</strong>
          <small>Total medicines issued</small>
        </article>
        <article>
          <span>Orders dispensed</span>
          <strong>{dispensedBillsCount}</strong>
          <small>Purchased & verified bills</small>
        </article>
        <article>
          <span>Follow-up plans</span>
          <strong>{visitGroups.filter((v) => Boolean(v.followUpPlan)).length}</strong>
          <small>Doctor scheduled reviews</small>
        </article>
      </section>

      {/* Controls Bar: Search, Filter, Sort */}
      <div className="portal-card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', flex: '1 1 360px' }}>
            <input
              type="search"
              placeholder="Search by medicine name, doctor, diagnosis, or follow-up…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                padding: '9px 14px',
                borderRadius: '8px',
                border: '1px solid #cfd6db',
                fontSize: '14px',
                flex: '1 1 260px',
                minWidth: '220px',
              }}
            />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              style={{
                padding: '9px 12px',
                borderRadius: '8px',
                border: '1px solid #cfd6db',
                fontSize: '14px',
                background: '#fff',
                cursor: 'pointer',
              }}
            >
              <option value="all">All dispensing statuses</option>
              <option value="dispensed">Dispensed & purchased only</option>
              <option value="pending">Awaiting pharmacy dispensing</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--ink-soft)' }}>Visit Sequence:</span>
            <button
              className={`button ${sortOrder === 'asc' ? 'button--primary' : 'button--outline'}`}
              style={{ padding: '8px 14px', fontSize: '13px', minHeight: '38px' }}
              onClick={() => setSortOrder('asc')}
              title="Track chronologically from earliest visit forward to present"
            >
              Past → Present (Serial)
            </button>
            <button
              className={`button ${sortOrder === 'desc' ? 'button--primary' : 'button--outline'}`}
              style={{ padding: '8px 14px', fontSize: '13px', minHeight: '38px' }}
              onClick={() => setSortOrder('desc')}
              title="Show latest visit first"
            >
              Present → Past (Newest)
            </button>
          </div>
        </div>
      </div>

      {/* Main Chronological Visit Records */}
      {loading ? (
        <div className="portal-card" style={{ padding: '40px', textAlign: 'center' }}>
          <div className="status-pill status-pill--active" style={{ marginBottom: '12px' }}>
            Loading prescriptions…
          </div>
          <p style={{ color: 'var(--ink-soft)', margin: 0 }}>
            Compiling your visit-by-visit medication and pharmacy history.
          </p>
        </div>
      ) : filteredVisits.length === 0 ? (
        <div className="portal-card" style={{ padding: '48px', textAlign: 'center' }}>
          <p style={{ fontSize: '16px', fontWeight: 600, marginBottom: '6px' }}>
            No prescription records found matching your filters.
          </p>
          <p style={{ color: 'var(--ink-soft)', fontSize: '14px', margin: 0 }}>
            Medications prescribed by doctors during consultations and pharmacy bills will appear here organized by visit.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {filteredVisits.map((group) => {
            return (
              <section
                key={group.visitKey}
                className="portal-card"
                style={{
                  padding: '24px',
                  border: '1px solid #cfd6db',
                  borderRadius: '14px',
                  background: '#fff',
                  boxShadow: '0 4px 14px rgba(15,23,42,0.04)',
                }}
              >
                {/* 1. Visit Header */}
                <header
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    flexWrap: 'wrap',
                    gap: '16px',
                    paddingBottom: '16px',
                    borderBottom: '1px solid #e2e8f0',
                    marginBottom: '20px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                    <div
                      style={{
                        background: '#f1f5f9',
                        color: '#0f172a',
                        fontWeight: 800,
                        padding: '10px 14px',
                        borderRadius: '10px',
                        textAlign: 'center',
                        border: '1px solid #cbd5e1',
                      }}
                    >
                      <span style={{ fontSize: '10px', textTransform: 'uppercase', display: 'block', color: '#64748b' }}>
                        Visit
                      </span>
                      <span style={{ fontSize: '20px' }}>#{String(group.serialNumber).padStart(2, '0')}</span>
                    </div>

                    <div>
                      <p className="eyebrow" style={{ margin: 0 }}>
                        {formatDate(group.dateStr)} · {group.departmentName}
                      </p>
                      <h2 style={{ margin: '3px 0', fontSize: '22px' }}>{group.doctorName}</h2>
                      <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>
                        {group.doctorSpecialty} · <strong>Reason:</strong> {group.reasonForVisit}
                      </p>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                    {group.status === 'dispensed' ? (
                      <span
                        style={{
                          background: '#ecfdf5',
                          color: '#065f46',
                          border: '1px solid #a7f3d0',
                          padding: '6px 12px',
                          borderRadius: '20px',
                          fontWeight: 700,
                          fontSize: '13px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                      >
                        ✓ Dispensed & Purchased
                      </span>
                    ) : group.status === 'billed' ? (
                      <span
                        style={{
                          background: '#eff6ff',
                          color: '#1e40af',
                          border: '1px solid #bfdbfe',
                          padding: '6px 12px',
                          borderRadius: '20px',
                          fontWeight: 700,
                          fontSize: '13px',
                        }}
                      >
                        ● Billed · Ready for Pickup
                      </span>
                    ) : (
                      <span
                        style={{
                          background: '#fffbeb',
                          color: '#92400e',
                          border: '1px solid #fde68a',
                          padding: '6px 12px',
                          borderRadius: '20px',
                          fontWeight: 700,
                          fontSize: '13px',
                        }}
                      >
                        ⏱ Prescribed · Sent to Pharmacy
                      </span>
                    )}

                    {group.appointment && (
                      <span className={`status-pill status-${group.appointment.status}`}>
                        Visit {group.appointment.status}
                      </span>
                    )}
                  </div>
                </header>

                {/* Working Diagnosis Bar */}
                <div
                  style={{
                    background: '#f8fafc',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    borderLeft: '4px solid #0a5a68',
                    marginBottom: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                  }}
                >
                  <strong style={{ fontSize: '13px', color: '#0a5a68', textTransform: 'uppercase' }}>
                    Clinical Diagnosis:
                  </strong>
                  <span style={{ fontSize: '14px', fontWeight: 600, color: '#1e293b' }}>
                    {group.diagnosis}
                  </span>
                </div>

                {/* 2. Prescribed Medications Table */}
                <div style={{ marginBottom: '24px' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ background: '#e0f2fe', color: '#0369a1', padding: '3px 8px', borderRadius: '6px', fontSize: '12px' }}>Rx</span>
                    Doctor Prescribed Medications
                  </h3>

                  <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '10px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                      <thead>
                        <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                          <th style={{ padding: '12px 16px', fontWeight: 700, color: '#475569' }}>Medicine Name</th>
                          <th style={{ padding: '12px 16px', fontWeight: 700, color: '#475569' }}>Dosage</th>
                          <th style={{ padding: '12px 16px', fontWeight: 700, color: '#475569' }}>Frequency</th>
                          <th style={{ padding: '12px 16px', fontWeight: 700, color: '#475569' }}>Duration</th>
                          <th style={{ padding: '12px 16px', fontWeight: 700, color: '#475569' }}>Instructions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {group.prescriptions.flatMap((rx) =>
                          rx.medications.map((med, idx) => (
                            <tr key={`${rx.id}-${idx}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                              <td style={{ padding: '12px 16px', fontWeight: 600, color: '#0f172a' }}>
                                {med.name}
                              </td>
                              <td style={{ padding: '12px 16px', color: '#334155' }}>
                                {med.dosage || 'As directed'}
                              </td>
                              <td style={{ padding: '12px 16px', color: '#334155' }}>
                                <span style={{ background: '#f1f5f9', padding: '3px 8px', borderRadius: '6px', fontSize: '13px' }}>
                                  {med.frequency || 'Daily'}
                                </span>
                              </td>
                              <td style={{ padding: '12px 16px', color: '#334155' }}>
                                {med.duration || 'Full course'}
                              </td>
                              <td style={{ padding: '12px 16px', color: '#475569', fontSize: '13px' }}>
                                {rx.instructions || 'Follow doctor directions'}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* 3. Pharmacy Dispensing & Purchased Bills Section */}
                <div style={{ marginBottom: '24px', background: '#fafafa', padding: '18px', borderRadius: '12px', border: '1px solid #e5e7eb' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
                    <h3 style={{ fontSize: '15px', fontWeight: 700, margin: 0, color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span>🏪</span> Pharmacy Dispensing & Purchased Bills
                    </h3>
                    {group.linkedBills.length > 0 && (
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#047857' }}>
                        {group.linkedBills.filter((b) => b.status === 'dispensed').length} order(s) dispensed & collected
                      </span>
                    )}
                  </div>

                  {group.linkedBills.length === 0 ? (
                    <div style={{ padding: '14px', background: '#fff', borderRadius: '8px', border: '1px dashed #cbd5e1' }}>
                      <p style={{ margin: 0, fontSize: '14px', color: '#4b5563' }}>
                        <strong>Awaiting Pharmacy Dispensing:</strong> This prescription is in queue with MedRitt Central Pharmacy.
                        Once the chemist validates stock and prepares the medicines, your itemized purchase bill will appear here.
                      </p>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {group.linkedBills.map((bill) => (
                        <div
                          key={bill.id}
                          style={{
                            background: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: '10px',
                            padding: '16px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            flexWrap: 'wrap',
                            gap: '14px',
                          }}
                        >
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                              <strong style={{ fontSize: '16px', color: '#0f172a' }}>
                                Invoice {bill.invoice_number}
                              </strong>
                              <span
                                style={{
                                  fontSize: '11px',
                                  textTransform: 'uppercase',
                                  fontWeight: 700,
                                  padding: '2px 8px',
                                  borderRadius: '12px',
                                  background: bill.status === 'dispensed' ? '#dcfce7' : '#dbeafe',
                                  color: bill.status === 'dispensed' ? '#15803d' : '#1d4ed8',
                                }}
                              >
                                {bill.status === 'dispensed' ? 'Dispensed & Paid' : 'Billed · Ready'}
                              </span>
                            </div>
                            <p style={{ margin: 0, fontSize: '13px', color: '#475569' }}>
                              Dispensed by: <strong>{bill.pharmacy.full_name}</strong> ·{' '}
                              {bill.items.length} item(s): {bill.items.map((it) => it.name).join(', ')}
                            </p>
                            {bill.dispensed_at && (
                              <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#64748b' }}>
                                Dispensed on: {formatDate(bill.dispensed_at)}
                              </p>
                            )}
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{ textAlign: 'right' }}>
                              <span style={{ fontSize: '11px', color: '#64748b', display: 'block', textTransform: 'uppercase' }}>
                                Total Amount
                              </span>
                              <strong style={{ fontSize: '18px', color: '#047857' }}>
                                {currency.format(bill.total)}
                              </strong>
                            </div>

                            <Link
                              to={`/medicine-bills/${bill.id}`}
                              className="button button--outline"
                              style={{ fontSize: '13px', padding: '8px 14px' }}
                            >
                              View Official Receipt →
                            </Link>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* 4. Doctor Follow-up Plan */}
                <div
                  style={{
                    background: '#eff6ff',
                    border: '1px solid #bfdbfe',
                    borderRadius: '10px',
                    padding: '16px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '14px',
                  }}
                >
                  <div>
                    <h4 style={{ margin: '0 0 4px', fontSize: '14px', color: '#1e40af', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span>📅</span> Doctor's Follow-up Plan & Advice
                    </h4>
                    <p style={{ margin: 0, fontSize: '14px', color: '#1e3a8a', fontWeight: 500 }}>
                      {group.followUpPlan || 'Routine recovery. Return for evaluation if symptoms persist or as guided by your physician.'}
                    </p>
                  </div>

                  <Link
                    to="/patient/book-appointment"
                    className="button button--primary"
                    style={{ fontSize: '13px', padding: '8px 14px' }}
                  >
                    Schedule Follow-up Visit →
                  </Link>
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
