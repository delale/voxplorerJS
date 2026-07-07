export interface UploadedTable {
  id: string
  name: string
  columns: string[]
  rowCount: number
}

export interface AudioFile {
  id: string
  name: string
  duration: number
  sampleRate: number
}

export interface DimensionalityReductionResult {
  dim1: number[]
  dim2: number[]
  dim3?: number[]
  method: 'pca' | 'umap' | 'tsne' | 'mds'
}

export interface VisualisationState {
  table?: UploadedTable
  audioFiles?: AudioFile[]
  reducedData?: DimensionalityReductionResult
  selectedIndices?: number[]
}
