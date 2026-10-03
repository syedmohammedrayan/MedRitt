import brandLogo from '../../../MedRittAI_Healthcare_Logo-removebg-preview.png';


type BrandLogoProps = {
  className?: string;
  variant?: 'default' | 'login';
};

export default function BrandLogo({ className = '', variant = 'default' }: BrandLogoProps) {
  const isLoginLogo = variant === 'login';

  return (
    <span className={`medora-logo ${className}`.trim()}>
      <img
        src={brandLogo}
        alt="MedRittAI logo"
        width={isLoginLogo ? 1536 : 608}
        height={isLoginLogo ? 1024 : 400}
        decoding="async"
      />
    </span>
  );
}
