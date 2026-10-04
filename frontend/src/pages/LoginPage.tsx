import { useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import BrandLogo from '../components/BrandLogo';
import ForgotPasswordModal from '../components/ForgotPasswordModal';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const { isAuthenticated, login, user } = useAuth();
  const navigate = useNavigate();

  if (isAuthenticated && user) {
    const home = user.role === 'patient'
      ? '/patient/dashboard'
      : user.role === 'lab_tech'
        ? '/lab/dashboard'
        : user.role === 'pharmacy'
          ? '/pharmacy/dashboard'
          : user.role === 'admin'
            ? '/admin/doctors'
            : '/doctor/dashboard';
    return <Navigate to={home} replace />;
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!username.trim() || !password) return;
    setLoading(true);
    setError('');
    try {
      await login({ username: username.trim(), password });
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to sign in. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <section className="login-editorial" aria-labelledby="login-wordmark" style={{ position: 'relative' }}>
        <Link to="/" style={{ position: 'absolute', top: '24px', left: '24px', color: '#fff', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: 600, opacity: 0.8, transition: 'opacity 0.2s' }} onMouseEnter={(e) => e.currentTarget.style.opacity = '1'} onMouseLeave={(e) => e.currentTarget.style.opacity = '0.8'}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
          Back to Home
        </Link>
        <div className="login-brand" style={{ marginTop: '30px' }}>
          <BrandLogo className="medritt-logo--auth" variant="login" />
          <span>Clinical imaging</span>
        </div>
        <div className="login-hero-copy">
          <p className="eyebrow eyebrow--light">Built for clinical pace</p>
          <h1 id="login-wordmark">Every scan.<br /><em>More clarity.</em></h1>
          <p>
            A focused workspace for image review, structured reporting, and clearer
            conversations with patients.
          </p>
        </div>
        <div className="login-index" aria-hidden="true">
          <span>01</span><span>Review</span><span>Report</span><span>Explain</span>
        </div>
      </section>

      <section className="login-form-panel">
        <div className="login-form-wrap">
          <p className="eyebrow">Secure hospital access</p>
          <h2>Welcome back.</h2>
          <p className="form-intro">Enter your credentials to open the diagnostic workspace.</p>

          <form onSubmit={handleSubmit} className="login-form">
            <label>
              <span>Username</span>
              <input
                type="text"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="Your username"
                autoComplete="username"
                autoFocus
              />
            </label>
            <label>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Password</span>
                <button type="button" onClick={() => setShowForgotPassword(true)} style={{ fontSize: '13px', color: 'var(--brand-primary)', textDecoration: 'none', background: 'none', border: 'none', padding: 0, cursor: 'pointer' }}>Forgot Password?</button>
              </div>
              <div className="password-input-wrapper" style={{ position: 'relative' }}>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Your password"
                  autoComplete="current-password"
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

            {error && <div className="form-error" role="alert">{error}</div>}

            <button className="button button--primary button--wide" type="submit" disabled={loading}>
              <span>{loading ? 'Opening workspace…' : 'Enter workspace'}</span>
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5" /></svg>
            </button>
          </form>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', fontSize: '0.875rem' }}>
            <p className="auth-switch" style={{ margin: 0 }}>New patient? <Link to="/register">Create an account</Link></p>
            <button type="button" onClick={() => setShowForgotPassword(true)} style={{ color: 'var(--brand-600)', textDecoration: 'none', background: 'none', border: 'none', padding: 0, cursor: 'pointer', fontSize: 'inherit' }}>Forgot password?</button>
          </div>
        </div>
        <p className="login-footer">MedRittAI Clinical Workspace · {new Date().getFullYear()}</p>
      </section>

      {showForgotPassword && (
        <ForgotPasswordModal onClose={() => setShowForgotPassword(false)} />
      )}
    </div>
  );
}
