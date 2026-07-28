import { render, screen } from "@testing-library/react";
import Home from "../app/page";

describe("Home page", () => {
  it("renders the hero heading", () => {
    render(<Home />);
    expect(screen.getByText(/Otoscopy classification that explains its reasoning\./i)).toBeInTheDocument();
  });

  it("renders the demo link", () => {
    render(<Home />);
    expect(screen.getByRole("link", { name: /Try the demo/i })).toBeInTheDocument();
  });
});
