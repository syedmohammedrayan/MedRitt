import React, { useState } from 'react';
import {
  forgotPasswordIdentify,
  forgotPasswordVerify,
  forgotPasswordReset
} from '../api/client';
import type { SecurityQuestion } from '../types';

interface Props {
  onClose: () => void;
}

export default function ForgotPasswordModal({ onClose }: Props) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [identifier, setIdentifier] = useState('');
  const [questions, setQuestions] = useState<SecurityQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleIdentify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!identifier.trim()) return;

    setLoading(true);
    setError('');

    try {
      const qs = await forgotPasswordIdentify({ identifier: identifier.trim() });
      if (!qs || qs.length === 0) {
        throw new Error("No security questions configured.");
      }
      setQuestions(qs);

      const initialAnswers: Record<string, string> = {};
      qs.forEach(q => { initialAnswers[q.question_id] = ''; });
      setAnswers(initialAnswers);

      setStep(2);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const formattedAnswers = Object.entries(answers).map(([question_id, answer]) => ({
      question_id,
      answer
    }));

    try {
      const res = await forgotPasswordVerify({
        identifier: identifier.trim(),
        answers: formattedAnswers
      });
      setResetToken(res.reset_token);
      setStep(3);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'The verification answers could not be confirmed.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    setError('');

    try {
      await forgotPasswordReset({
        reset_token: resetToken,
        new_password: newPassword
      });
      setSuccess(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to reset password.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(20, 24, 27, 0.65)',
      backdropFilter: 'blur(4px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        backgroundColor: 'var(--panel)',
        borderRadius: '16px',
        padding: '32px',
        width: '100%',
        maxWidth: '440px',
        boxShadow: '0 24px 60px -26px rgba(20, 24, 27, 0.55)',
        position: 'relative',
        maxHeight: '90vh',
        overflowY: 'auto'
      }}>
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            fontSize: '24px',
            color: 'var(--ink-soft)',
            lineHeight: 1
          }}
          aria-label="Close"
        >
          ×
        </button>

        {success ? (
          <>
            <h2 style={{ fontSize: '24px', marginBottom: '8px' }}>Password Reset Complete</h2>
            <p className="form-intro" style={{ marginBottom: '24px' }}>Your password has been successfully reset. You can now log in with your new credentials.</p>
            <button className="button button--primary button--wide" onClick={onClose}>
              <span>Return to Login</span>
            </button>
          </>
        ) : (
          <>
            {step === 1 && (
              <>
                <h2 style={{ fontSize: '24px', marginBottom: '8px' }}>Forgot Password</h2>
                <p className="form-intro" style={{ marginBottom: '24px' }}>Enter your username or email address to recover your account.</p>
                <form onSubmit={handleIdentify} className="login-form">
                  <label>
                    <span>Username or Email</span>
                    <input
                      type="text"
                      value={identifier}
                      onChange={(e) => setIdentifier(e.target.value)}
                      placeholder="Enter your identifier"
                      required
                      autoFocus
                    />
                  </label>
                  {error && <div className="form-error" role="alert">{error}</div>}
                  <button className="button button--primary button--wide" type="submit" disabled={loading || !identifier}>
                    <span>{loading ? 'Finding Account…' : 'Continue'}</span>
                  </button>
                </form>
              </>
            )}

            {step === 2 && (
              <>
                <h2 style={{ fontSize: '24px', marginBottom: '8px' }}>Verify Your Identity</h2>
                <p className="form-intro" style={{ marginBottom: '24px' }}>Please answer the security questions you previously configured.</p>
                <form onSubmit={handleVerify} className="login-form">
                  {questions.map((q, index) => (
                    <label key={q.question_id}>
                      <span>Question {index + 1}: {q.question}</span>
                      <input
                        type="text"
                        value={answers[q.question_id] || ''}
                        onChange={(e) => setAnswers({ ...answers, [q.question_id]: e.target.value })}
                        placeholder="Your answer"
                        required
                      />
                    </label>
                  ))}
                  {error && <div className="form-error" role="alert">{error}</div>}
                  <button className="button button--primary button--wide" type="submit" disabled={loading}>
                    <span>{loading ? 'Verifying…' : 'Verify Answers'}</span>
                  </button>
                  <button type="button" className="button button--outline button--wide" onClick={() => setStep(1)} style={{ marginTop: '12px' }}>
                    <span>Back</span>
                  </button>
                </form>
              </>
            )}

            {step === 3 && (
              <>
                <h2 style={{ fontSize: '24px', marginBottom: '8px' }}>Set New Password</h2>
                <p className="form-intro" style={{ marginBottom: '24px' }}>Create a new password for your account.</p>
                <form onSubmit={handleReset} className="login-form">
                  <label>
                    <span>New Password</span>
                    <div className="password-input-wrapper" style={{ position: 'relative' }}>
                      <input
                        type={showPassword ? "text" : "password"}
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="••••••••"
                        required
                        minLength={6}
                        style={{ width: '100%', paddingRight: '40px' }}
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px' }}
                        aria-label={showPassword ? "Hide password" : "Show password"}
                      >
                        {showPassword ? "👁️" : "👁️‍🗨️"}
                      </button>
                    </div>
                  </label>
                  <label>
                    <span>Confirm Password</span>
                    <div className="password-input-wrapper" style={{ position: 'relative' }}>
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="••••••••"
                        required
                        minLength={6}
                        style={{ width: '100%', paddingRight: '40px' }}
                      />
                      <button
                        type="button"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px' }}
                        aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                      >
                        {showConfirmPassword ? "👁️" : "👁️‍🗨️"}
                      </button>
                    </div>
                  </label>
                  {error && <div className="form-error" role="alert">{error}</div>}
                  <button className="button button--primary button--wide" type="submit" disabled={loading}>
                    <span>{loading ? 'Resetting…' : 'Reset Password'}</span>
                  </button>
                </form>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
