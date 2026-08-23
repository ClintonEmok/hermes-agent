import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Ephemeral pane persistence (#92818): preview/route tile panes are mirrored
// from live tab lists that die with the process. Persisting them 1:1 made the
// stored layout describe panes that could never come back, so every restart
// re-arranged a hand-built split. The LIVE tree must keep its tiles; only the
// STORED copy drops them.

const TREE_KEY = 'hermes.desktop.layoutTree.v2'

const treeWithTiles = {
  type: 'split',
  id: 'root',
  orientation: 'row',
  weights: [2, 1],
  children: [
    { type: 'group', id: 'g-main', panes: ['workspace', 'preview-tile:file:a'], active: 'preview-tile:file:a' },
    {
      type: 'group',
      id: 'g-right',
      panes: ['terminal', 'route-tile:/skills'],
      active: 'route-tile:/skills'
    }
  ]
}

describe('ephemeral panes are stripped from the persisted layout tree', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const store = await import('@/components/pane-shell/tree/store')
    return store
  }

  it('persist drops ephemeral panes but keeps the live tree intact', async () => {
    const store = await setup()

    store.$layoutTree.set(treeWithTiles as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as {
      children?: Array<{ panes?: string[] }>
    }

    // Stored copy: tiles gone, real chrome stays.
    expect(persisted.children?.[0]?.panes).toEqual(['workspace'])
    expect(persisted.children?.[1]?.panes).toEqual(['terminal'])

    // Live tree: untouched — the preview is still on screen.
    const live = store.$layoutTree.get()
    expect(live).not.toBeNull()
    expect(JSON.stringify(live)).toContain('preview-tile:file:a')
  })

  it('a group holding ONLY ephemeral panes is pruned from the stored copy', async () => {
    const store = await setup()

    store.$layoutTree.set({
      type: 'split',
      id: 'root',
      orientation: 'row',
      weights: [1, 1],
      children: [
        { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' },
        { type: 'group', id: 'g-tiles', panes: ['preview-tile:url:x', 'route-tile:/y'], active: 'preview-tile:url:x' }
      ]
    } as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!)

    // normalize collapses the now-empty right group into the root.
    expect(JSON.stringify(persisted)).not.toContain('g-tiles')
    expect(JSON.stringify(persisted)).toContain('workspace')
  })

  it('the active pane falls back to a surviving sibling when the active was ephemeral', async () => {
    const store = await setup()

    store.$layoutTree.set(treeWithTiles as never)
    store.persistTree()

    const persisted = JSON.parse(window.localStorage.getItem(TREE_KEY)!) as {
      children?: Array<{ active?: string; panes?: string[] }>
    }

    expect(persisted.children?.[0]?.active).toBe('workspace')
    expect(persisted.children?.[1]?.active).toBe('terminal')
  })

  it('stripEphemeralPanes leaves a tree without ephemeral panes unchanged (same reference)', () => {
    void import('@/components/pane-shell/tree/store').then(async () => {
      const { stripEphemeralPanes } = await import('@/components/pane-shell/tree/store')
      const plain = {
        type: 'group',
        id: 'g',
        panes: ['workspace', 'terminal'],
        active: 'workspace'
      }

      expect(stripEphemeralPanes(plain as never)).toBe(plain)
    })
  })
})
