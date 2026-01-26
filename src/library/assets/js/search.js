/**
 * Glenn Bot Knowledge Library - Client-Side Search
 *
 * Uses Fuse.js for fuzzy search with a pre-built index generated at build time.
 * Features:
 * - Lazy-loading of search index on first interaction
 * - 300ms debounce for input
 * - Fuzzy search with configurable weights
 * - Results displayed as cards
 */

(function () {
  "use strict";

  // Configuration
  const CONFIG = {
    indexPath: "/assets/js/search-index.json",
    debounceMs: 300,
    maxResults: 50,
    minQueryLength: 2,
    // Fuse.js options with field weights
    fuseOptions: {
      keys: [
        { name: "title", weight: 1.0 },
        { name: "keywords", weight: 0.8 },
        { name: "summary", weight: 0.6 },
        { name: "themes", weight: 0.4 },
        { name: "content_type", weight: 0.2 },
      ],
      threshold: 0.4, // Lower = more strict matching
      includeScore: true,
      ignoreLocation: true,
      minMatchCharLength: 2,
    },
  };

  // State
  let searchIndex = null;
  let fuse = null;
  let isLoading = false;
  let debounceTimer = null;

  /**
   * Load the search index from the JSON file.
   * Caches the result for subsequent searches.
   *
   * @returns {Promise<Array>} The search index items
   */
  async function loadSearchIndex() {
    if (searchIndex !== null) {
      return searchIndex;
    }

    if (isLoading) {
      // Wait for existing load to complete
      return new Promise((resolve) => {
        const checkLoaded = setInterval(() => {
          if (searchIndex !== null) {
            clearInterval(checkLoaded);
            resolve(searchIndex);
          }
        }, 50);
      });
    }

    isLoading = true;

    try {
      const response = await fetch(CONFIG.indexPath);
      if (!response.ok) {
        throw new Error(`Failed to load search index: ${response.status}`);
      }
      const data = await response.json();
      searchIndex = data.items || [];

      // Initialize Fuse.js with the loaded index
      if (typeof Fuse !== "undefined") {
        fuse = new Fuse(searchIndex, CONFIG.fuseOptions);
      } else {
        console.error("Fuse.js not loaded");
      }

      return searchIndex;
    } catch (error) {
      console.error("Error loading search index:", error);
      return [];
    } finally {
      isLoading = false;
    }
  }

  /**
   * Perform a search and return results.
   *
   * @param {string} query - The search query
   * @returns {Promise<Array>} Search results
   */
  async function search(query) {
    query = query.trim();

    if (query.length < CONFIG.minQueryLength) {
      return [];
    }

    await loadSearchIndex();

    if (!fuse) {
      // Fallback: simple substring search if Fuse.js failed to load
      return searchIndex
        .filter(
          (item) =>
            item.title.toLowerCase().includes(query.toLowerCase()) ||
            item.summary.toLowerCase().includes(query.toLowerCase())
        )
        .slice(0, CONFIG.maxResults);
    }

    const results = fuse.search(query, { limit: CONFIG.maxResults });
    return results.map((result) => result.item);
  }

  /**
   * Debounce function to limit how often a function is called.
   *
   * @param {Function} func - Function to debounce
   * @param {number} wait - Milliseconds to wait
   * @returns {Function} Debounced function
   */
  function debounce(func, wait) {
    return function (...args) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => func.apply(this, args), wait);
    };
  }

  /**
   * Get the badge class for a content type.
   *
   * @param {string} contentType - The content type
   * @returns {string} CSS class for the badge
   */
  function getBadgeClass(contentType) {
    return `badge badge-${contentType}`;
  }

  /**
   * Format content type for display (e.g., "web_content" -> "Web Content").
   *
   * @param {string} contentType - The content type
   * @returns {string} Formatted content type
   */
  function formatContentType(contentType) {
    return contentType
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  /**
   * Create HTML for a search result card.
   *
   * @param {Object} item - Search result item
   * @returns {string} HTML string for the card
   */
  function createResultCard(item) {
    const badgeClass = getBadgeClass(item.content_type);
    const typeDisplay = formatContentType(item.content_type);

    return `
      <article class="card">
        <a href="${item.url}" class="card-link">
          <div class="card-image card-image--default card-image--placeholder">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
          </div>
          <div class="card-content">
            <span class="${badgeClass}">${typeDisplay}</span>
            <h3 class="card-title">${escapeHtml(item.title)}</h3>
            <p class="card-summary">${escapeHtml(item.summary)}</p>
          </div>
        </a>
      </article>
    `;
  }

  /**
   * Escape HTML special characters to prevent XSS.
   *
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Render search results to the results container.
   *
   * @param {Array} results - Search results
   * @param {string} query - The search query
   */
  function renderResults(results, query) {
    const resultsContainer = document.getElementById("search-results");
    const resultsCount = document.getElementById("results-count");
    const noResults = document.getElementById("no-results");
    const searchQuery = document.getElementById("search-query-display");

    if (!resultsContainer) return;

    // Update query display
    if (searchQuery) {
      searchQuery.textContent = query;
    }

    // Update results count
    if (resultsCount) {
      resultsCount.textContent = `${results.length} result${results.length !== 1 ? "s" : ""}`;
    }

    // Handle no results
    if (results.length === 0 && query.length >= CONFIG.minQueryLength) {
      resultsContainer.innerHTML = "";
      if (noResults) {
        noResults.style.display = "block";
      }
      return;
    }

    // Hide no results message
    if (noResults) {
      noResults.style.display = "none";
    }

    // Render result cards
    if (results.length > 0) {
      resultsContainer.innerHTML = results.map(createResultCard).join("");
    } else {
      resultsContainer.innerHTML = "";
    }
  }

  /**
   * Handle search input event.
   *
   * @param {Event} event - Input event
   */
  async function handleSearchInput(event) {
    const query = event.target.value;
    const results = await search(query);
    renderResults(results, query);
  }

  /**
   * Handle search form submission.
   *
   * @param {Event} event - Submit event
   */
  function handleSearchSubmit(event) {
    event.preventDefault();
    const input = event.target.querySelector('input[type="search"]');
    if (input) {
      const query = input.value.trim();
      if (query.length >= CONFIG.minQueryLength) {
        // Navigate to search page with query
        window.location.href = `/search/?q=${encodeURIComponent(query)}`;
      }
    }
  }

  /**
   * Initialize search functionality on page load.
   */
  function initSearch() {
    // Initialize header search form
    const headerSearchForm = document.querySelector(".search-form");
    if (headerSearchForm) {
      headerSearchForm.addEventListener("submit", handleSearchSubmit);

      // Pre-load index on focus for faster first search
      const searchInput = headerSearchForm.querySelector(".search-input");
      if (searchInput) {
        searchInput.addEventListener("focus", loadSearchIndex, { once: true });
      }
    }

    // Initialize search page if we're on it
    const searchPageInput = document.getElementById("search-page-input");
    if (searchPageInput) {
      // Set up debounced search
      const debouncedSearch = debounce(handleSearchInput, CONFIG.debounceMs);
      searchPageInput.addEventListener("input", debouncedSearch);

      // Check for query parameter and perform initial search
      const urlParams = new URLSearchParams(window.location.search);
      const initialQuery = urlParams.get("q");
      if (initialQuery) {
        searchPageInput.value = initialQuery;
        // Trigger search immediately for initial query
        handleSearchInput({ target: searchPageInput });
      }

      // Focus the search input
      searchPageInput.focus();
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSearch);
  } else {
    initSearch();
  }

  // Expose search function globally for potential external use
  window.LibrarySearch = {
    search: search,
    loadIndex: loadSearchIndex,
  };
})();

/**
 * Glenn Bot Knowledge Library - Sort & Pagination
 *
 * Provides sort functionality (Recent, A-Z, By Type) and pagination
 * for the All Content page.
 */

(function () {
  "use strict";

  // Configuration
  const PAGINATION_CONFIG = {
    itemsPerPage: 50,
  };

  // State
  let allItems = [];
  let displayedItems = [];
  let currentSort = "recent";
  let currentPage = 1;
  let isLoading = false;

  /**
   * Sort items by most recently added (created_at descending).
   *
   * @param {Array} items - Items to sort
   * @returns {Array} Sorted items
   */
  function sortByRecent(items) {
    return [...items].sort((a, b) => {
      const dateA = new Date(a.created_at || 0);
      const dateB = new Date(b.created_at || 0);
      return dateB - dateA;
    });
  }

  /**
   * Sort items alphabetically by title (A-Z).
   *
   * @param {Array} items - Items to sort
   * @returns {Array} Sorted items
   */
  function sortByTitle(items) {
    return [...items].sort((a, b) => {
      const titleA = (a.title || "").toLowerCase();
      const titleB = (b.title || "").toLowerCase();
      return titleA.localeCompare(titleB);
    });
  }

  /**
   * Sort items by content type, then alphabetically within each type.
   *
   * @param {Array} items - Items to sort
   * @returns {Array} Sorted items
   */
  function sortByType(items) {
    return [...items].sort((a, b) => {
      const typeA = a.content_type || "";
      const typeB = b.content_type || "";

      // First sort by type
      const typeCompare = typeA.localeCompare(typeB);
      if (typeCompare !== 0) {
        return typeCompare;
      }

      // Then by title within the same type
      const titleA = (a.title || "").toLowerCase();
      const titleB = (b.title || "").toLowerCase();
      return titleA.localeCompare(titleB);
    });
  }

  /**
   * Apply the current sort to items.
   *
   * @param {Array} items - Items to sort
   * @param {string} sortType - Sort type: "recent", "alpha", or "type"
   * @returns {Array} Sorted items
   */
  function applySort(items, sortType) {
    switch (sortType) {
      case "alpha":
        return sortByTitle(items);
      case "type":
        return sortByType(items);
      case "recent":
      default:
        return sortByRecent(items);
    }
  }

  /**
   * Get the badge class for a content type.
   *
   * @param {string} contentType - The content type
   * @returns {string} CSS class for the badge
   */
  function getBadgeClass(contentType) {
    return `badge badge-${contentType}`;
  }

  /**
   * Format content type for display (e.g., "web_content" -> "Web Content").
   *
   * @param {string} contentType - The content type
   * @returns {string} Formatted content type
   */
  function formatContentType(contentType) {
    return contentType
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  /**
   * Escape HTML special characters to prevent XSS.
   *
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Create HTML for an item card.
   *
   * @param {Object} item - Item data
   * @returns {string} HTML string for the card
   */
  function createItemCard(item) {
    const badgeClass = getBadgeClass(item.content_type);
    const typeDisplay = formatContentType(item.content_type);
    const imageClass =
      item.content_type === "book" ? "card-image--book" : "card-image--default";

    // Use cover image if available, otherwise placeholder
    const imageContent = item.cover_image_url
      ? `<img src="${escapeHtml(item.cover_image_url)}" alt="${escapeHtml(item.title)}">`
      : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
        </svg>`;

    const placeholderClass = item.cover_image_url ? "" : "card-image--placeholder";

    return `
      <article class="card" data-content-type="${escapeHtml(item.content_type)}">
        <a href="${escapeHtml(item.url)}" class="card-link">
          <div class="card-image ${imageClass} ${placeholderClass}">
            ${imageContent}
          </div>
          <div class="card-content">
            <span class="${badgeClass}">${typeDisplay}</span>
            <h3 class="card-title">${escapeHtml(item.title)}</h3>
            <p class="card-summary">${escapeHtml(item.summary)}</p>
          </div>
        </a>
      </article>
    `;
  }

  /**
   * Create type header for grouped display.
   *
   * @param {string} contentType - The content type
   * @returns {string} HTML string for the type header
   */
  function createTypeHeader(contentType) {
    const typeDisplay = formatContentType(contentType);
    return `<h2 class="type-group-header" data-type="${escapeHtml(contentType)}">${typeDisplay}</h2>`;
  }

  /**
   * Render items to the grid container.
   *
   * @param {Array} items - Items to render
   * @param {boolean} append - If true, append to existing content; otherwise replace
   */
  function renderItems(items, append = false) {
    const gridContainer = document.getElementById("all-content-grid");
    if (!gridContainer) return;

    if (!append) {
      gridContainer.innerHTML = "";
    }

    // For "type" sort, add headers between groups
    if (currentSort === "type") {
      let currentType = null;
      const fragment = document.createDocumentFragment();
      const tempDiv = document.createElement("div");

      items.forEach((item) => {
        if (item.content_type !== currentType) {
          currentType = item.content_type;
          // Type header as full-width element
          tempDiv.innerHTML = createTypeHeader(currentType);
          const header = tempDiv.firstElementChild;
          if (header) {
            fragment.appendChild(header);
          }
        }
        tempDiv.innerHTML = createItemCard(item);
        const card = tempDiv.firstElementChild;
        if (card) {
          fragment.appendChild(card);
        }
      });

      gridContainer.appendChild(fragment);
    } else {
      // Simple rendering without type headers
      const html = items.map(createItemCard).join("");
      if (append) {
        gridContainer.insertAdjacentHTML("beforeend", html);
      } else {
        gridContainer.innerHTML = html;
      }
    }
  }

  /**
   * Update the results count display.
   *
   * @param {number} shown - Number of items currently shown
   * @param {number} total - Total number of items
   */
  function updateResultsCount(shown, total) {
    const countEl = document.getElementById("all-content-count");
    if (countEl) {
      countEl.textContent = `Showing ${shown} of ${total} items`;
    }
  }

  /**
   * Update the load more button visibility.
   *
   * @param {boolean} hasMore - Whether there are more items to load
   */
  function updateLoadMoreButton(hasMore) {
    const loadMoreBtn = document.getElementById("load-more-btn");
    if (loadMoreBtn) {
      loadMoreBtn.style.display = hasMore ? "inline-block" : "none";
    }
  }

  /**
   * Display the current page of items.
   */
  function displayCurrentPage() {
    const startIndex = 0;
    const endIndex = currentPage * PAGINATION_CONFIG.itemsPerPage;
    displayedItems = allItems.slice(startIndex, endIndex);

    renderItems(displayedItems);
    updateResultsCount(displayedItems.length, allItems.length);
    updateLoadMoreButton(displayedItems.length < allItems.length);
  }

  /**
   * Handle sort selection change.
   *
   * @param {Event} event - Change event
   */
  function handleSortChange(event) {
    const newSort = event.target.value;
    if (newSort === currentSort) return;

    currentSort = newSort;
    currentPage = 1;

    // Re-sort and display
    allItems = applySort(allItems, currentSort);
    displayCurrentPage();

    // Save sort preference to session storage
    try {
      sessionStorage.setItem("library-sort", currentSort);
    } catch (e) {
      // Ignore storage errors
    }
  }

  /**
   * Handle load more button click.
   */
  function handleLoadMore() {
    if (displayedItems.length >= allItems.length) return;

    currentPage++;
    displayCurrentPage();
  }

  /**
   * Load library data from the JSON file.
   *
   * @returns {Promise<Array>} Library items
   */
  async function loadLibraryData() {
    if (isLoading) return [];

    isLoading = true;

    try {
      // Try to load from the library data file
      const response = await fetch("/assets/js/search-index.json");
      if (!response.ok) {
        throw new Error(`Failed to load library data: ${response.status}`);
      }
      const data = await response.json();
      return data.items || [];
    } catch (error) {
      console.error("Error loading library data:", error);
      return [];
    } finally {
      isLoading = false;
    }
  }

  /**
   * Initialize the All Content page functionality.
   */
  async function initAllContentPage() {
    const gridContainer = document.getElementById("all-content-grid");
    if (!gridContainer) return; // Not on the All Content page

    const sortSelect = document.getElementById("sort-select");
    const loadMoreBtn = document.getElementById("load-more-btn");

    // Load saved sort preference
    try {
      const savedSort = sessionStorage.getItem("library-sort");
      if (savedSort && ["recent", "alpha", "type"].includes(savedSort)) {
        currentSort = savedSort;
        if (sortSelect) {
          sortSelect.value = currentSort;
        }
      }
    } catch (e) {
      // Ignore storage errors
    }

    // Set up event listeners
    if (sortSelect) {
      sortSelect.addEventListener("change", handleSortChange);
    }

    if (loadMoreBtn) {
      loadMoreBtn.addEventListener("click", handleLoadMore);
    }

    // Load and display items
    const items = await loadLibraryData();
    allItems = applySort(items, currentSort);
    displayCurrentPage();
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAllContentPage);
  } else {
    initAllContentPage();
  }

  // Expose functions globally for potential external use
  window.LibraryContent = {
    sort: applySort,
    loadData: loadLibraryData,
  };
})();
