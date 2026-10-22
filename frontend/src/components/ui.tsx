import { useState, type ButtonHTMLAttributes, type ComponentProps, type InputHTMLAttributes, type ReactNode } from 'react'
import { Link, type LinkProps } from 'react-router-dom'
import { toggleTheme } from '../lib/theme'
import PixelCritter from './PixelCritter'

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

/* ------------------------------------------------------------------ card -- */

export function Card({ children, className, padded = true }: { children: ReactNode; className?: string; padded?: boolean }) {
  return <section className={cx('rounded-2xl border-2 border-hairline bg-surface', padded && 'p-5', className)}>{children}</section>
}

export function CardHeader({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h2 className="text-base font-extrabold text-ink">{title}</h2>
        {description && <p className="mt-0.5 text-sm text-ink-muted">{description}</p>}
      </div>
      {action}
    </div>
  )
}

/* ---------------------------------------------------------------- button -- */

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'lg'

// chunky buttons: primary / secondary / danger sit on a darker edge (.press in index.css) and sink onto it when pressed
const VARIANTS: Record<Variant, string> = {
  primary: 'press bg-brand text-white [--edge:var(--brand-dark)] hover:brightness-[1.06]',
  secondary: 'press border-2 border-hairline bg-surface text-brand [--edge:var(--border)] hover:bg-surface-hover',
  ghost: 'bg-transparent text-ink-secondary hover:bg-surface-hover hover:text-ink',
  danger: 'press border-2 border-danger-border bg-surface text-danger [--edge:var(--danger-border)] hover:bg-danger-bg',
}
const SIZES: Record<Size, string> = {
  sm: 'h-9 px-3.5 text-xs gap-1.5 [--lift:3px]',
  md: 'h-11 px-5 text-[13px] gap-2',
  lg: 'h-12 px-6 text-sm gap-2',
}
const buttonClass = (variant: Variant, size: Size, className?: string) =>
  cx(
    'inline-flex select-none items-center justify-center whitespace-nowrap rounded-2xl font-extrabold uppercase tracking-[0.06em]',
    'disabled:cursor-not-allowed disabled:opacity-50',
    VARIANTS[variant],
    SIZES[size],
    className,
  )

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size }) {
  return <button className={buttonClass(variant, size, className)} {...rest} />
}

export function ButtonLink({ variant = 'secondary', size = 'md', className, ...rest }: LinkProps & { variant?: Variant; size?: Size }) {
  return <Link className={buttonClass(variant, size, className)} {...rest} />
}

/* ----------------------------------------------------------------- input -- */

const FIELD =
  'w-full rounded-2xl border-2 border-hairline bg-surface-muted px-4 text-[15px] font-semibold text-ink placeholder:font-medium placeholder:text-ink-muted ' +
  'transition-colors focus:border-brand-border focus:bg-surface focus:outline-none ' +
  'disabled:cursor-not-allowed disabled:text-ink-muted'

export function Input({ className, ...rest }: ComponentProps<'input'>) {
  return <input className={cx(FIELD, 'h-12', className)} {...rest} />
}

/** A labelled field. `quiet` keeps the label for screen readers only (the input's placeholder says what it is). */
export function Field({ label, hint, htmlFor, quiet = false, children }: { label: ReactNode; hint?: ReactNode; htmlFor?: string; quiet?: boolean; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className={quiet ? 'sr-only' : 'block text-sm font-bold text-ink-secondary'}>{label}</label>
      {children}
      {hint && <p className="text-xs text-ink-muted">{hint}</p>}
    </div>
  )
}

/** A password field you can read back (off by default). */
export function PasswordInput({ className, ...rest }: Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>) {
  const [visible, setVisible] = useState(false)
  return (
    <div className="relative">
      <Input {...rest} type={visible ? 'text' : 'password'} className={cx('pr-10', className)} />
      <button
        type="button"
        onClick={() => setVisible((v) => !v)}
        tabIndex={-1}
        aria-label={visible ? 'Hide password' : 'Show password'}
        title={visible ? 'Hide password' : 'Show password'}
        className="absolute right-1 top-0 flex h-12 w-11 items-center justify-center rounded-r-2xl text-ink-muted transition-colors hover:text-ink"
      >
        <Icon>
          {visible ? (
            <>
              <path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c7 0 10 8 10 8a18.5 18.5 0 0 1-2.16 3.19M6.61 6.61A18.6 18.6 0 0 0 2 12s3 8 10 8a9.1 9.1 0 0 0 5.39-1.61" />
              <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
              <path d="m2 2 20 20" />
            </>
          ) : (
            <>
              <path d="M2 12s3-8 10-8 10 8 10 8-3 8-10 8-10-8-10-8Z" />
              <circle cx="12" cy="12" r="3" />
            </>
          )}
        </Icon>
      </button>
    </div>
  )
}

/* ------------------------------------------------------------------ misc -- */

export function Icon({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={cx('h-4 w-4 shrink-0', className)}>
      {children}
    </svg>
  )
}

export function PageHeader({ title, description, actions }: { title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-2xl font-extrabold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 text-sm text-ink-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

export function Alert({ tone = 'danger', children }: { tone?: 'danger' | 'brand'; children: ReactNode }) {
  return (
    <div
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cx(
        'rounded-2xl border-2 px-4 py-3 text-sm font-semibold',
        tone === 'danger' ? 'border-danger-border bg-danger-bg text-danger' : 'border-brand-border bg-brand-soft text-ink',
      )}
    >
      {children}
    </div>
  )
}

export function EmptyState({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-2xl border-2 border-dashed border-hairline-strong bg-surface-muted px-6 py-10 text-center">
      <div className="mx-auto mb-3 w-16"><PixelCritter seed="sleepy-empty" size={64} sleepy /></div>
      <p className="text-base font-extrabold text-ink">{title}</p>
      {description && <p className="mx-auto mt-1 max-w-sm text-sm text-ink-muted">{description}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <span role="status" aria-label="Loading" className={cx('inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent', className)} />
}

export function Loading() {
  return (
    <div className="flex flex-1 items-center justify-center py-24">
      <Spinner className="text-ink-muted" />
    </div>
  )
}

export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd className={cx('inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-hairline-strong bg-surface px-1 font-mono text-[11px] font-bold normal-case tracking-normal text-ink-muted', className)}>
      {children}
    </kbd>
  )
}

/** Sun/moon switch: shows the theme you'd switch to. Both icons render; CSS on [data-theme] picks one. */
export function ThemeToggle({ className }: { className?: string }) {
  return (
    <button
      type="button"
      onClick={toggleTheme}
      title="Toggle light and dark theme"
      aria-label="Toggle light and dark theme"
      className={cx('rounded-xl p-2 text-ink-muted transition-colors hover:bg-surface-hover hover:text-ink', className)}
    >
      <Icon>
        <path className="theme-when-light" d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        <g className="theme-when-dark">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </g>
      </Icon>
    </button>
  )
}

export function Stat({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="rounded-2xl border-2 border-hairline bg-surface px-4 py-3.5">
      <div className="text-2xl font-extrabold tracking-tight text-ink tabular-nums">{value}</div>
      <div className="mt-0.5 text-xs font-semibold text-ink-muted">{label}</div>
    </div>
  )
}
