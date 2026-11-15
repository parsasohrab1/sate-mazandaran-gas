{{/*
Common labels for weld-defect resources.
*/}}
{{- define "weld-defect.labels" -}}
app.kubernetes.io/name: weld-defect
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: sate-mazandaran-gas
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "weld-defect.selectorLabels" -}}
app.kubernetes.io/name: weld-defect
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name.
*/}}
{{- define "weld-defect.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (printf "%s-api" .Release.Name) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end }}
