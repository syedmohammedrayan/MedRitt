import { Link } from 'react-router-dom';
import brandLogoDefault from '../../../MedRittAI_Healthcare_Logo-removebg-preview.png';

type BrandLogoProps = {
  className?: string;
  variant?: 'default' | 'login';
};

export default function BrandLogo({ className = '', variant = 'default' }: BrandLogoProps) {
  const isLoginLogo = variant === 'login';

  return (
    <a href="/" className={`medritt-logo ${className}`.trim()} style={{ textDecoration: 'none', display: 'inline-flex' }}>
      <img
        src={isLoginLogo ? "/logo.png" : brandLogoDefault}
        alt="MedRittAI logo"
        width={isLoginLogo ? 1536 : 608}
        height={isLoginLogo ? 1024 : 400}
        decoding="async"
        style={{ maxWidth: '100%', height: 'auto', objectFit: 'contain' }}
      />
    </a>
  );
}
