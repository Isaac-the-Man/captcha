import { Home } from "~/pages/Home";
import type { Route } from "../+types/root";
import { ThemeProvider, CssBaseline } from "@mui/material";
import { defaultTheme } from "~/theme/default";


export function meta({ }: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function HomeRoute() {
  return (
  <ThemeProvider theme={defaultTheme}>
    <CssBaseline />
    <Home />
  </ThemeProvider>);
}
