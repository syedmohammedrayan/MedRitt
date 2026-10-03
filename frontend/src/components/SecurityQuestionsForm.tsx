import React, { useEffect, useState } from 'react';
import { getMySecurityQuestions, getSecurityQuestionsBank, setupSecurityQuestions } from '../api/client';
import type { SecurityQuestion } from '../types';

export default function SecurityQuestionsForm() {
  const [bank, setBank] = useState<SecurityQuestion[]>([]);
  const [selectedQuestions, setSelectedQuestions] = useState<{ question_id: string; answer: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    Promise.all([
      getSecurityQuestionsBank().catch(() => [] as SecurityQuestion[]),
      getMySecurityQuestions().catch(() => [] as SecurityQuestion[])
    ])
      .then(([bankData, myData]) => {
        setBank(bankData);
        if (myData && myData.length > 0) {
          setSelectedQuestions(myData.map(q => ({ question_id: q.question_id, answer: '' })));
        } else {
          setSelectedQuestions([
            { question_id: '', answer: '' },
            { question_id: '', answer: '' },
          ]);
        }
      })
      .catch(() => setError('Failed to load security questions.'))
      .finally(() => setLoading(false));
  }, []);

  const handleSelect = (index: number, question_id: string) => {
    const next = [...selectedQuestions];
    next[index].question_id = question_id;
    setSelectedQuestions(next);
  };

  const handleAnswer = (index: number, answer: string) => {
    const next = [...selectedQuestions];
    next[index].answer = answer;
    setSelectedQuestions(next);
  };

  const addQuestion = () => {
    if (selectedQuestions.length < Math.min(bank.length || 5, 5)) {
      setSelectedQuestions([...selectedQuestions, { question_id: '', answer: '' }]);
    }
  };

  const removeQuestion = (index: number) => {
    if (selectedQuestions.length > 1) {
      const next = [...selectedQuestions];
      next.splice(index, 1);
      setSelectedQuestions(next);
    }
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError('');
    setMessage('');

    try {
      const qIds = selectedQuestions.map((q) => q.question_id).filter(Boolean);
      if (qIds.length === 0) {
        throw new Error('Please select at least 1 security question.');
      }
      if (new Set(qIds).size !== qIds.length) {
        throw new Error('Please select unique questions.');
      }

      for (const q of selectedQuestions) {
        if (!q.question_id || !q.answer.trim()) {
           throw new Error('Please choose a question and provide an answer for all slots.');
        }
      }

      await setupSecurityQuestions({ questions: selectedQuestions });
      setMessage('Security questions saved successfully.');

      setSelectedQuestions(selectedQuestions.map(q => ({ ...q, answer: '' })));

    } catch (err: any) {
      setError(err instanceof Error ? err.message : err.response?.data?.detail || 'Could not save security questions.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div style={{ padding: '12px 0', fontSize: '12px', color: 'var(--ink-soft)' }}>Loading security settings...</div>;

  return (
    <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--line)' }}>
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'none',
          border: 'none',
          padding: 0,
          cursor: 'pointer',
        }}
      >
        <div style={{ textAlign: 'left' }}>
          <strong style={{ fontSize: '13px', fontWeight: 600, color: 'var(--ink)' }}>Password Recovery</strong>
          <p style={{ fontSize: '11px', color: 'var(--ink-soft)', margin: '2px 0 0' }}>
            Configure security questions for account recovery.
          </p>
        </div>
        <span style={{
          fontSize: '10px',
          color: 'var(--ink-soft)',
          transform: isExpanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }}>
          ▼
        </span>
      </button>

      {isExpanded && (
        <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {error && <p className="account-panel__error">{error}</p>}
          {message && <p style={{ color: 'var(--teal)', fontSize: '12px', margin: 0 }}>{message}</p>}

          {selectedQuestions.map((q, index) => {
            const availableQuestions = bank.filter(
              (bq) => !selectedQuestions.some((sq, sqIdx) => sqIdx !== index && sq.question_id === bq.question_id)
            );

            return (
              <div key={index} style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                padding: '10px',
                backgroundColor: 'var(--paper)',
                borderRadius: '8px',
                border: '1px solid var(--line)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--ink-soft)' }}>
                    Question {index + 1}
                  </span>
                  {selectedQuestions.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeQuestion(index)}
                      style={{ fontSize: '10px', color: 'var(--heat-red)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
                    >
                      Remove
                    </button>
                  )}
                </div>
                <select
                  required
                  style={{
                    fontSize: '12px',
                    padding: '7px 8px',
                    borderRadius: '6px',
                    border: '1px solid var(--line)',
                    backgroundColor: '#fff',
                    color: 'var(--ink)',
                  }}
                  value={q.question_id}
                  onChange={(e) => handleSelect(index, e.target.value)}
                >
                  <option value="" disabled>Select a question...</option>
                  {availableQuestions.map((bq) => (
                    <option key={bq.question_id} value={bq.question_id}>
                      {bq.question}
                    </option>
                  ))}
                </select>
                <input
                  required
                  type="text"
                  placeholder="Your answer"
                  style={{
                    fontSize: '12px',
                    padding: '7px 8px',
                    borderRadius: '6px',
                    border: '1px solid var(--line)',
                    backgroundColor: '#fff',
                    color: 'var(--ink)',
                  }}
                  value={q.answer}
                  onChange={(e) => handleAnswer(index, e.target.value)}
                />
              </div>
            );
          })}

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
            {selectedQuestions.length < 5 ? (
              <button
                type="button"
                onClick={addQuestion}
                style={{ fontSize: '11px', fontWeight: 600, color: 'var(--teal)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
              >
                + Add Question
              </button>
            ) : (
              <span style={{ fontSize: '11px', color: 'var(--ink-soft)' }}>Maximum 5 questions</span>
            )}

            <button
              type="button"
              onClick={submit}
              disabled={saving}
              className="button button--primary"
              style={{ fontSize: '12px', padding: '8px 14px' }}
            >
              {saving ? 'Saving...' : 'Save Questions'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
