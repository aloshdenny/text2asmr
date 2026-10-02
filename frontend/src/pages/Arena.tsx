import { AppLayout } from '../components/Layouts'
import { ButtonLink, EmptyState, PageHeader } from '../components/ui'

export default function Arena() {
  return (
    <AppLayout>
      <PageHeader title="Arena" />
      <EmptyState title="The Arena is on its way" description="Check back soon." action={<ButtonLink to="/dashboard">Back to your dashboard</ButtonLink>} />
    </AppLayout>
  )
}
