import argparse
import csv
import os
import shutil
import time
from pathlib import Path
from typing import Dict

from selenium import webdriver
from selenium.common.exceptions import (
    JavascriptException,
    NoSuchDriverException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService


GRID_SELECTOR = "div.ag-center-cols-container"
ROW_SELECTOR = "div.ag-center-cols-container div[role='row'][row-id]"


EXTRACT_ROWS_JS = r"""
const getCellText = (cell) => {
  if (!cell) return "";
  const labeled = cell.querySelector("[aria-label]");
  if (labeled) {
    const value = labeled.getAttribute("aria-label");
    if (value && value.trim()) return value.trim();
  }
  return (cell.textContent || "").trim();
};

const parseRowIndex = (row) => {
  const raw = row.getAttribute("row-index") || row.getAttribute("aria-rowindex") || "";
  const value = parseInt(raw, 10);
  return Number.isFinite(value) ? value : -1;
};

const rows = Array.from(document.querySelectorAll("div.ag-center-cols-container div[role='row'][row-id]"));
return rows
  .map((row) => {
    const caseCell = row.querySelector("div[col-id='ticketnumber']");
    const contactCell = row.querySelector("div[col-id='primarycontactid']");
    const countryCell = row.querySelector("div[col-id='msdfm_countryid']");
    return {
      row_id: row.getAttribute("row-id") || "",
      row_index: parseRowIndex(row),
      case_number: getCellText(caseCell),
      contact: getCellText(contactCell),
      country_region: getCellText(countryCell),
    };
  })
  .filter((r) => r.row_id);
"""


HORIZONTAL_SCROLL_JS = r"""
const direction = arguments[0];
const selectors = [
  ".ag-body-horizontal-scroll-viewport",
  ".ag-center-cols-viewport"
];

for (const selector of selectors) {
  const el = document.querySelector(selector);
  if (!el) continue;
  const max = Math.max(0, el.scrollWidth - el.clientWidth);
  if (max <= 0) continue;
  const target = direction === "right" ? max : 0;
  const before = el.scrollLeft;
  el.scrollLeft = target;
  el.dispatchEvent(new Event("scroll", { bubbles: true }));
  return { selector, before, after: el.scrollLeft, max };
}
return null;
"""


VERTICAL_SCROLL_JS = r"""
const step = arguments[0];
const selectors = [
  ".ag-body-vertical-scroll-viewport",
  ".ag-body-viewport",
  ".ag-center-cols-viewport"
];

const states = [];
for (const selector of selectors) {
  const elements = Array.from(document.querySelectorAll(selector));
  for (const el of elements) {
    const max = Math.max(0, el.scrollHeight - el.clientHeight);
    if (max <= 0) continue;
    const before = el.scrollTop;
    el.scrollTop = Math.min(max, before + step);
    el.dispatchEvent(new Event("scroll", { bubbles: true }));
    try {
      el.dispatchEvent(new WheelEvent("wheel", { deltaY: step, bubbles: true }));
    } catch (e) {
      // no-op
    }
    states.push({
      selector: selector,
      before: before,
      after: el.scrollTop,
      max: max
    });
  }
}
return states;
"""


BOTTOM_NUDGE_JS = r"""
const step = arguments[0];
const selectors = [
  ".ag-body-vertical-scroll-viewport",
  ".ag-body-viewport",
  ".ag-center-cols-viewport"
];

const states = [];
for (const selector of selectors) {
  const elements = Array.from(document.querySelectorAll(selector));
  for (const el of elements) {
    const max = Math.max(0, el.scrollHeight - el.clientHeight);
    if (max <= 0) continue;
    const before = el.scrollTop;
    const retreat = Math.max(0, max - Math.max(120, Math.min(step, 450)));
    el.scrollTop = retreat;
    el.dispatchEvent(new Event("scroll", { bubbles: true }));
    el.scrollTop = max;
    el.dispatchEvent(new Event("scroll", { bubbles: true }));
    try {
      el.dispatchEvent(new WheelEvent("wheel", { deltaY: step, bubbles: true }));
    } catch (e) {
      // no-op
    }
    states.push({
      selector: selector,
      before: before,
      retreat: retreat,
      after: el.scrollTop,
      max: max
    });
  }
}
return states;
"""


GRID_PROGRESS_JS = r"""
const rows = Array.from(document.querySelectorAll("div.ag-center-cols-container div[role='row'][row-id]"));
const indices = rows
  .map((row) => parseInt(row.getAttribute("row-index") || row.getAttribute("aria-rowindex") || "-1", 10))
  .filter((n) => Number.isFinite(n) && n >= 0);

const maxIndex = indices.length ? Math.max(...indices) : -1;
const minIndex = indices.length ? Math.min(...indices) : -1;
return {
  visible_count: rows.length,
  min_index: minIndex,
  max_index: maxIndex
};
"""


TOTAL_ROWS_JS = r"""
const parseCount = (text) => {
  if (!text) return null;
  const m = text.match(/Rows:\s*([0-9,]+)/i);
  if (!m) return null;
  const n = parseInt(m[1].replace(/,/g, ""), 10);
  return Number.isFinite(n) ? n : null;
};

const selectors = [
  ".ag-status-bar",
  ".ag-status-bar-left",
  ".ag-root-wrapper",
  "body"
];

for (const selector of selectors) {
  const nodes = Array.from(document.querySelectorAll(selector));
  for (const node of nodes) {
    const value = parseCount(node.innerText || node.textContent || "");
    if (value !== null) return value;
  }
}
return null;
"""


def _resolve_existing_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    expanded = os.path.expandvars(path_value)
    candidate = Path(expanded).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return None


def find_msedgedriver(explicit_driver_path: str | None = None) -> str | None:
    candidates = [
        explicit_driver_path,
        os.environ.get("MSEDGEDRIVER_PATH"),
        shutil.which("msedgedriver"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedgedriver.exe",
    ]
    for item in candidates:
        resolved = _resolve_existing_path(item)
        if resolved:
            return resolved

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            for found in winget_root.glob("Microsoft.EdgeDriver_*\\msedgedriver.exe"):
                return str(found.resolve())
    return None


def find_edge_binary() -> str | None:
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for item in candidates:
        resolved = _resolve_existing_path(item)
        if resolved:
            return resolved
    return None


def build_driver(
    headless: bool,
    user_data_dir: str | None,
    driver_path: str | None,
) -> webdriver.Edge:
    options = EdgeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    if headless:
        options.add_argument("--headless=new")
    if user_data_dir:
        options.add_argument(f"--user-data-dir={user_data_dir}")

    edge_binary = find_edge_binary()
    if edge_binary:
        options.binary_location = edge_binary

    resolved_driver = find_msedgedriver(driver_path)
    if resolved_driver:
        print(f"Using local EdgeDriver: {resolved_driver}")
        service = EdgeService(executable_path=resolved_driver)
        return webdriver.Edge(service=service, options=options)

    print("Local EdgeDriver not found. Selenium Manager will attempt to download it.")
    try:
        return webdriver.Edge(options=options)
    except NoSuchDriverException as exc:
        raise RuntimeError(
            "Unable to start Edge WebDriver.\n"
            "Install EdgeDriver once with:\n"
            "  winget install -e --id Microsoft.EdgeDriver\n"
            "or run this script with:\n"
            "  --driver-path \"C:\\path\\to\\msedgedriver.exe\""
        ) from exc


def _switch_to_grid_context_recursive(driver: webdriver.Edge, max_depth: int) -> bool:
    if driver.find_elements(By.CSS_SELECTOR, GRID_SELECTOR):
        return True
    if max_depth <= 0:
        return False

    frames = driver.find_elements(By.CSS_SELECTOR, "iframe")
    for frame in frames:
        try:
            driver.switch_to.frame(frame)
        except WebDriverException:
            continue
        if _switch_to_grid_context_recursive(driver, max_depth - 1):
            return True
        driver.switch_to.parent_frame()
    return False


def switch_to_grid_context(driver: webdriver.Edge, timeout_sec: int, max_depth: int = 4) -> bool:
    end = time.time() + timeout_sec
    while time.time() < end:
        driver.switch_to.default_content()
        if _switch_to_grid_context_recursive(driver, max_depth=max_depth):
            return True
        time.sleep(0.75)
    driver.switch_to.default_content()
    return False


def wait_for_rows(driver: webdriver.Edge, timeout_sec: int) -> None:
    end = time.time() + timeout_sec
    while time.time() < end:
        if driver.find_elements(By.CSS_SELECTOR, ROW_SELECTOR):
            return
        time.sleep(0.25)
    raise TimeoutError("Grid found, but no rows appeared within timeout.")


def ensure_grid_ready(driver: webdriver.Edge, timeout_sec: int = 30) -> bool:
    if not switch_to_grid_context(driver, timeout_sec=timeout_sec):
        return False
    try:
        wait_for_rows(driver, timeout_sec=timeout_sec)
        return True
    except TimeoutError:
        return False


def merge_rows(seen: Dict[str, Dict[str, str]], rows: list[dict]) -> tuple[int, int]:
    new_rows = 0
    filled_fields = 0

    for row in rows:
        row_id = (row.get("row_id") or "").strip()
        if not row_id:
            continue

        case_number = (row.get("case_number") or "").strip()
        contact = (row.get("contact") or "").strip()
        country_region = (row.get("country_region") or "").strip()
        try:
            row_index = int(row.get("row_index", -1))
        except (TypeError, ValueError):
            row_index = -1

        existing = seen.get(row_id)
        if not existing:
            seen[row_id] = {
                "row_id": row_id,
                "row_index": row_index,
                "case_number": case_number,
                "contact": contact,
                "country_region": country_region,
            }
            new_rows += 1
            continue

        if existing["row_index"] < 0 and row_index >= 0:
            existing["row_index"] = row_index
        if not existing["case_number"] and case_number:
            existing["case_number"] = case_number
            filled_fields += 1
        if not existing["contact"] and contact:
            existing["contact"] = contact
            filled_fields += 1
        if not existing["country_region"] and country_region:
            existing["country_region"] = country_region
            filled_fields += 1

    return new_rows, filled_fields


def collect_all_rows(
    driver: webdriver.Edge,
    scroll_step: int,
    settle_ms: int,
    max_idle_loops: int,
    max_loops: int,
    recovery_timeout_sec: int,
    max_missing_viewport_loops: int,
    locked_window_handle: str | None = None,
    max_bottom_probe_loops: int = 25,
    target_rows: int | None = None,
) -> Dict[str, Dict[str, str]]:
    # Dynamics grid is slow to refresh after virtual scroll; keep at least 1s settle.
    effective_settle_ms = max(1000, settle_ms)

    seen: Dict[str, Dict[str, str]] = {}
    idle_loops = 0
    stuck_scroll_loops = 0
    missing_viewport_loops = 0
    best_visible_max_index = -1
    best_scroll_max = -1.0
    bottom_probe_loops = 0
    effective_target_rows = target_rows if (target_rows and target_rows > 0) else None

    for loop in range(1, max_loops + 1):
        if locked_window_handle:
            try:
                handles = driver.window_handles
                if locked_window_handle in handles:
                    if driver.current_window_handle != locked_window_handle:
                        driver.switch_to.window(locked_window_handle)
                elif handles:
                    driver.switch_to.window(handles[0])
                    locked_window_handle = driver.current_window_handle
            except WebDriverException:
                pass

        if not driver.find_elements(By.CSS_SELECTOR, ROW_SELECTOR):
            print(f"[loop {loop}] rows not visible in current context; attempting grid recovery...")
            if not ensure_grid_ready(driver, timeout_sec=recovery_timeout_sec):
                missing_viewport_loops += 1
                if missing_viewport_loops >= max_missing_viewport_loops:
                    print("Stop condition reached: grid rows unavailable for too many loops.")
                    break
                time.sleep(effective_settle_ms / 1000.0)
                continue

        try:
            # Pass 1: scroll horizontally to left to capture Case Number/Contact.
            driver.execute_script(HORIZONTAL_SCROLL_JS, "left")
            left_rows = driver.execute_script(EXTRACT_ROWS_JS) or []

            # Pass 2: scroll horizontally to right to capture Country/Region if column is virtualized.
            driver.execute_script(HORIZONTAL_SCROLL_JS, "right")
            right_rows = driver.execute_script(EXTRACT_ROWS_JS) or []

            # Reset horizontal scroll for the next loop.
            driver.execute_script(HORIZONTAL_SCROLL_JS, "left")
        except JavascriptException as exc:
            print(f"[loop {loop}] JS read failed ({exc.__class__.__name__}); attempting grid recovery...")
            if ensure_grid_ready(driver, timeout_sec=recovery_timeout_sec):
                time.sleep(effective_settle_ms / 1000.0)
                continue
            raise RuntimeError(f"Failed to read grid rows via JavaScript: {exc}") from exc

        before_count = len(seen)
        new_left, fill_left = merge_rows(seen, left_rows)
        new_right, fill_right = merge_rows(seen, right_rows)
        after_count = len(seen)
        new_rows = new_left + new_right
        filled_fields = fill_left + fill_right

        if effective_target_rows is None:
            try:
                detected_total = driver.execute_script(TOTAL_ROWS_JS)
                if detected_total is not None:
                    detected_total = int(detected_total)
                    if detected_total > 0:
                        effective_target_rows = detected_total
                        print(f"Detected target row count from page: {effective_target_rows}")
            except (JavascriptException, ValueError, TypeError):
                pass

        if effective_target_rows is not None and after_count >= effective_target_rows:
            print(
                f"Stop condition reached: extracted {after_count} rows, "
                f"meeting target {effective_target_rows}."
            )
            break

        try:
            progress = driver.execute_script(GRID_PROGRESS_JS) or {}
        except JavascriptException as exc:
            print(f"[loop {loop}] JS progress failed ({exc.__class__.__name__}); attempting grid recovery...")
            if ensure_grid_ready(driver, timeout_sec=recovery_timeout_sec):
                time.sleep(effective_settle_ms / 1000.0)
                continue
            raise RuntimeError(f"Failed to read grid progress via JavaScript: {exc}") from exc

        visible_max_index = int(progress.get("max_index", -1))
        index_progress = visible_max_index > best_visible_max_index
        if index_progress:
            best_visible_max_index = visible_max_index

        try:
            states = driver.execute_script(VERTICAL_SCROLL_JS, scroll_step) or []
        except JavascriptException as exc:
            print(f"[loop {loop}] JS scroll failed ({exc.__class__.__name__}); attempting grid recovery...")
            if ensure_grid_ready(driver, timeout_sec=recovery_timeout_sec):
                time.sleep(effective_settle_ms / 1000.0)
                continue
            raise RuntimeError(f"Failed to scroll grid via JavaScript: {exc}") from exc

        if not states:
            print(f"[loop {loop}] scroll viewport not found; attempting to reacquire grid context...")
            if ensure_grid_ready(driver, timeout_sec=recovery_timeout_sec):
                missing_viewport_loops += 1
                if missing_viewport_loops >= max_missing_viewport_loops:
                    print("Stop condition reached: AG Grid viewport missing for too many loops.")
                    break
                idle_loops += 1
                time.sleep(effective_settle_ms / 1000.0)
                continue
            missing_viewport_loops += 1
            if missing_viewport_loops >= max_missing_viewport_loops:
                print("Stop condition reached: unable to reacquire AG Grid viewport for too many loops.")
                break
            time.sleep(effective_settle_ms / 1000.0)
            continue

        missing_viewport_loops = 0

        moved = any(float(s.get("after", 0)) > float(s.get("before", 0)) for s in states)
        at_bottom = all(float(s.get("after", 0)) >= float(s.get("max", 0)) - 2 for s in states)
        current_scroll_max = max(float(s.get("max", 0.0)) for s in states) if states else 0.0
        scroll_extent_progress = current_scroll_max > best_scroll_max + 1
        if scroll_extent_progress:
            best_scroll_max = current_scroll_max

        # Avoid keyboard/click fallback actions to keep the script non-intrusive.

        had_data_progress = (
            (after_count > before_count)
            or (filled_fields > 0)
            or index_progress
            or scroll_extent_progress
        )
        idle_loops = 0 if had_data_progress else idle_loops + 1
        stuck_scroll_loops = 0 if moved else stuck_scroll_loops + 1
        bottom_probe_loops = 0 if had_data_progress else bottom_probe_loops

        if loop % 10 == 0 or had_data_progress:
            print(
                f"[loop {loop}] unique_rows={after_count} new_rows={new_rows} "
                f"filled={filled_fields} visible_max_row_index={visible_max_index} "
                f"idle_loops={idle_loops} stuck_scroll_loops={stuck_scroll_loops}"
            )

        if at_bottom and not had_data_progress and bottom_probe_loops < max_bottom_probe_loops:
            bottom_probe_loops += 1
            try:
                driver.execute_script(BOTTOM_NUDGE_JS, scroll_step)
            except JavascriptException:
                pass
            if bottom_probe_loops % 5 == 0:
                print(
                    f"[loop {loop}] bottom probe {bottom_probe_loops}/{max_bottom_probe_loops} "
                    "while waiting for next chunk..."
                )
            time.sleep(effective_settle_ms / 1000.0)
            continue

        if idle_loops >= max_idle_loops and stuck_scroll_loops >= 3 and at_bottom:
            print(
                "Stop condition reached: no new data, no scroll movement, and viewport at bottom. "
                f"Detected visible row index up to {best_visible_max_index}."
            )
            break
        if idle_loops >= (max_idle_loops * 2) and stuck_scroll_loops >= 3:
            print("Stop condition reached: persistent no-progress loops.")
            break

        time.sleep(effective_settle_ms / 1000.0)
    else:
        print(f"Reached max_loops={max_loops}. Stopping collection.")

    return seen


def write_csv(path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    ordered_rows = sorted(
        rows.values(),
        key=lambda r: (
            int(r.get("row_index", -1)),
            (r.get("case_number") or ""),
            (r.get("row_id") or ""),
        ),
    )

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Case Number", "Contact", "Country/Region"],
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "Case Number": row["case_number"],
                    "Contact": row["contact"],
                    "Country/Region": row["country_region"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Case Number, Contact, Country/Region from Dynamics 365 AG Grid."
    )
    parser.add_argument(
        "--url",
        default="https://onesupport.crm.dynamics.com/main.aspx?appid=101acb62-8d00-eb11-a813-000d3a8b3117",
        help="Dynamics page URL that contains the cases table.",
    )
    parser.add_argument(
        "--output",
        default="cases_extract.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless.",
    )
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help="Optional Edge user data directory to reuse login session.",
    )
    parser.add_argument(
        "--driver-path",
        default=None,
        help="Optional full path to msedgedriver.exe.",
    )
    parser.add_argument(
        "--scroll-step",
        type=int,
        default=900,
        help="Pixels to scroll each iteration.",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=3000,
        help="Wait time after each scroll (milliseconds, minimum effective wait is 1000ms).",
    )
    parser.add_argument(
        "--max-idle-loops",
        type=int,
        default=25,
        help="Stop if no new rows are captured for this many loops.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=10000,
        help="Hard upper limit on scroll loops to avoid infinite runs.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1000,
        help="Timeout waiting for grid and initial rows.",
    )
    parser.add_argument(
        "--post-enter-delay-sec",
        type=int,
        default=5,
        help="Extra wait after pressing Enter to let slow Dynamics pages settle.",
    )
    parser.add_argument(
        "--recovery-timeout-sec",
        type=int,
        default=60,
        help="Timeout for each grid recovery attempt during scrolling.",
    )
    parser.add_argument(
        "--max-missing-viewport-loops",
        type=int,
        default=30,
        help="Stop only after this many loops with missing grid/viewport.",
    )
    parser.add_argument(
        "--max-bottom-probe-loops",
        type=int,
        default=25,
        help="Near-bottom retry loops to trigger loading the next server chunk before stopping.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Optional hard stop when extracted row count reaches this value (e.g. 1421).",
    )
    parser.add_argument(
        "--no-manual-wait",
        action="store_true",
        help="Skip manual Enter prompt after opening the page.",
    )
    args = parser.parse_args()

    driver = build_driver(
        headless=args.headless,
        user_data_dir=args.user_data_dir,
        driver_path=args.driver_path,
    )
    try:
        driver.get(args.url)
        if not args.no_manual_wait:
            input("Log in and open the cases table, then press Enter here...")
        if args.post_enter_delay_sec > 0:
            print(f"Waiting {args.post_enter_delay_sec}s for grid/view/filter to fully settle...")
            time.sleep(args.post_enter_delay_sec)
        locked_window_handle = driver.current_window_handle

        if not switch_to_grid_context(driver, timeout_sec=args.timeout_sec):
            raise TimeoutError("Could not find the table grid (or containing iframe).")

        wait_for_rows(driver, timeout_sec=args.timeout_sec)
        rows = collect_all_rows(
            driver,
            scroll_step=args.scroll_step,
            settle_ms=args.settle_ms,
            max_idle_loops=args.max_idle_loops,
            max_loops=args.max_loops,
            recovery_timeout_sec=args.recovery_timeout_sec,
            max_missing_viewport_loops=args.max_missing_viewport_loops,
            locked_window_handle=locked_window_handle,
            max_bottom_probe_loops=args.max_bottom_probe_loops,
            target_rows=args.target_rows,
        )
        case_count = sum(1 for r in rows.values() if r.get("case_number"))
        country_count = sum(1 for r in rows.values() if r.get("country_region"))
        output_path = Path(args.output).expanduser().resolve()
        write_csv(output_path, rows)
        print(
            f"Saved {len(rows)} row ids ({case_count} with case number, "
            f"{country_count} with country) to: {output_path}"
        )
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
